import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from wikijs2rag.enrich import Enricher, authority, load_topics
from wikijs2rag.linker import Linker
from wikijs2rag.llm import LLM, LLMError, parse_json
from wikijs2rag.staging import StagedPage, Staging, entity_aliases

URL = "http://llm.test/v1/chat/completions"


@pytest.fixture
def staging(tmp_path: Path) -> Iterator[Staging]:
    with Staging(f"sqlite:///{tmp_path / 's.db'}") as s:
        yield s


def _reply(content: object) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


PAGE = StagedPage("1", "en", "tools/laser", "Laser Cutter", "public", "u", "Text body", "h", tags=["lasers"])


@respx.mock
def test_text_is_cached(staging: Staging) -> None:
    route = respx.post(URL).mock(return_value=_reply("<think>hmm</think> Answer"))
    llm = LLM(staging, "http://llm.test/v1", "gen", api_key="dummy-key")
    assert llm.text("t", "sys", "prompt") == "Answer"
    assert llm.text("t", "sys", "prompt") == "Answer"
    assert route.call_count == 1
    assert route.calls.last.request.headers["Authorization"] == "Bearer dummy-key"
    assert json.loads(route.calls.last.request.content)["temperature"] == 0
    llm.close()


@respx.mock
@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, text="down"), httpx.Response(200, json={"choices": []}), _reply(None)],
)
def test_errors(staging: Staging, response: httpx.Response) -> None:
    respx.post(URL).mock(return_value=response)
    with pytest.raises(LLMError):
        LLM(staging, "http://llm.test/v1", "gen").text("t", "s", "p")


@respx.mock
def test_transport_error(staging: Staging) -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("no"))
    with pytest.raises(LLMError, match="failed"):
        LLM(staging, "http://llm.test/v1", "gen").text("t", "s", "p")


@respx.mock
def test_vision_cached_by_image(staging: Staging) -> None:
    route = respx.post(URL).mock(return_value=_reply("A diagram"))
    llm = LLM(staging, "http://llm.test/v1", "gen", vision_model="vis")
    assert llm.describe_image(b"img", "image/png", "describe", "") == "A diagram"
    assert llm.describe_image(b"img", "image/png", "describe", "") == "A diagram"
    assert route.call_count == 1
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "vis"
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    with pytest.raises(LLMError, match="VISION_MODEL"):
        LLM(staging, "http://llm.test/v1", "gen").describe_image(b"x", "image/png", "p", "")


def test_parse_json() -> None:
    assert parse_json('Sure: ```json\n{"a": [1]}\n```') == {"a": [1]}
    assert parse_json('["x", {"y": 1}]') == ["x", {"y": 1}]
    with pytest.raises(LLMError):
        parse_json("no json {here")


def test_fallbacks_without_llm() -> None:
    e = Enricher(None, ["lasers", "safety"])
    assert e.page_kind(PAGE) == "equipment"
    assert (
        e.page_kind(StagedPage("2", "en", "board/minutes-2024", "Minutes", "member", "u", "", "h")) == "meeting-notes"
    )
    assert e.page_kind(StagedPage("3", "en", "x", "Hello", "public", "u", "", "h")) == "other"
    assert e.summary(PAGE) == "Text body"
    assert e.page_topics(PAGE, "") == ["lasers"]
    assert (
        e.context(PAGE, "", "Laser Cutter > Use", "x")
        == 'From section "Laser Cutter > Use" of the wiki page "Laser Cutter".'
    )
    with pytest.raises(LLMError):
        e.exclusion("t", "x")
    assert (authority("policy"), authority("training"), authority("equipment")) == (5, 5, 4)


@respx.mock
def test_llm_answers_and_bad_answers(staging: Staging) -> None:
    replies = iter(['{"kind": "nonsense"}', "not json", "LLM summary", '{"topics": "oops"}'])
    respx.post(URL).mock(side_effect=lambda _r: _reply(next(replies)))
    e = Enricher(LLM(staging, "http://llm.test/v1", "gen"), ["lasers"])
    assert e.page_kind(PAGE) == "equipment"  # unknown kind: heuristic
    assert e.page_kind(StagedPage("9", "en", "y", "Y", "public", "u", "other", "h")) == "other"  # not JSON
    assert e.summary(PAGE) == "LLM summary"
    assert e.page_topics(PAGE, "s") == []  # a string, not a list: no valid slugs


@respx.mock
def test_exclusion_screens_long_pages_in_pieces(staging: Staging) -> None:
    verdicts = iter(['{"exclude": false}', '{"exclude": "true", "reason": "secrets"}'])
    route = respx.post(URL).mock(side_effect=lambda _r: _reply(next(verdicts)))
    e = Enricher(LLM(staging, "http://llm.test/v1", "gen"), [])
    assert e.exclusion("T", "word " * 3000) == "secrets"
    assert route.call_count == 2
    respx.post(URL).mock(return_value=_reply('{"nope": 1}'))
    with pytest.raises(LLMError, match="verdict"):
        e.exclusion("T", "other text")


def test_load_topics(tmp_path: Path) -> None:
    f = tmp_path / "t.txt"
    f.write_text("# comment\nlasers  # inline\n\nsewing\n")
    assert load_topics(str(f)) == ["lasers", "sewing"]
    assert "3d-printing" in load_topics(None)


@respx.mock
def test_linker(staging: Staging) -> None:
    with staging.begin() as conn:
        conn.execute(
            entity_aliases.insert(),
            [
                {"entity_type": "asset", "entity_id": "a1", "alias": "Big  Laser"},
                {"entity_type": "asset", "entity_id": "a2", "alias": "printer"},
                {"entity_type": "asset", "entity_id": "a3", "alias": "printer"},
                {"entity_type": "location", "entity_id": "l1", "alias": "ab"},  # too short, ignored
                {"entity_type": "unknown", "entity_id": "u1", "alias": "mystery"},
            ],
        )
    plain = Linker(staging, None)
    assert plain.link("Use the big\nlaser, not the printer. ab mystery").payload()["asset_ids"] == ["a1"]
    replies = iter(['{"entity": "asset:a3"}', '{"entity": "asset:zzz"}', "nope"])
    respx.post(URL).mock(side_effect=lambda _r: _reply(next(replies)))
    smart = Linker(staging, LLM(staging, "http://llm.test/v1", "gen"))
    assert smart.link("the printer one").payload()["asset_ids"] == ["a3"]
    assert smart.link("the printer two").payload()["asset_ids"] == []
    assert smart.link("the printer three").payload()["asset_ids"] == []
    links, roles = plain.link_page("equipment", "Big Laser", "tools/big-laser", [])
    assert roles == [("asset", "a1", "how-to-use")]
    assert plain.section(links) is None


def test_linker_without_aliases(staging: Staging) -> None:
    assert Linker(staging, None).link("anything").payload()["asset_ids"] == []
