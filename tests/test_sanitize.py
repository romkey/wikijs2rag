import pytest

from wikijs2rag.sanitize import REDACTED, scan_secrets, scrub_network_identifiers

# Fake secrets, assembled so no scanner flags this file.
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
GH = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
KEY = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIabc\n-----END RSA " + "PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (f"key {AWS} here", "aws-access-key"),
        (f"token {GH}", "github-token"),
        (KEY, "private-key"),
        ("xoxb-" + "1234567890-abcdefghij", "slack-token"),
        ("https://hooks.slack.com/services/T000/B000/XXXX", "slack-webhook"),
        ("db: postgres://user:" + "s3cr3tpw@db.example/x", "url-credentials"),
        ("ota_password: 'Hj7#kd92Lq'", "assigned-secret"),
        ("API_KEY=" + "9f8e7d6c5b4a", "assigned-secret"),
    ],
)
def test_finds_and_redacts(text: str, rule: str) -> None:
    redacted, hits = scan_secrets(text)
    assert [h.rule for h in hits] == [rule]
    assert REDACTED in redacted


@pytest.mark.parametrize(
    "text",
    [
        "password: !secret wifi_password",
        "api_key: ${API_KEY}",
        "password = changeme",
        "password: <your password>",
        "token: aaaaaaaa",  # low entropy
        "Ask a board member for the password.",
    ],
)
def test_ignores_placeholders(text: str) -> None:
    assert scan_secrets(text) == (text, [])


def test_line_numbers_and_only_secret_group_redacted() -> None:
    redacted, hits = scan_secrets(f"line one\nwifi_password: Zq8#kLm2pX9w\n{AWS}")
    assert [(h.rule, h.line) for h in hits] == [("assigned-secret", 2), ("aws-access-key", 3)]
    assert redacted == f"line one\nwifi_password: {REDACTED}\n{REDACTED}"


def test_network_identifiers_only_in_code_blocks() -> None:
    md = (
        "Server at 10.0.0.1.\n\n```yaml\nip: 192.168.1.20/24\nmac: AA:BB:CC:DD:EE:FF\nhost: door.local\n"
        "v6: fe80::1:2:3:4\nentity: binary_sensor.front_door\n```\n"
    )
    out = scrub_network_identifiers(md)
    assert out.startswith("Server at 10.0.0.1.")
    for placeholder in ("<IP>", "<MAC>", "<HOSTNAME>", "<ENTITY_ID>"):
        assert placeholder in out
    assert "192.168" not in out
    assert "front_door" not in out
    assert "fe80" not in out
