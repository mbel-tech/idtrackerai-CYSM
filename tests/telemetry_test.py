"""Usage analytics are opt-in in this fork, and version checks must not crash."""

from unittest import mock

import pytest

from idtrackerai.utils import telemetry
from idtrackerai.utils.telemetry import ComparisonResult, parse_release


@pytest.mark.parametrize(
    "version, expected",
    [
        ("6.0.15", (6, 0, 15)),
        ("6.0.15a0", (6, 0, 15)),
        ("6.0.15a0+detectron2.1", (6, 0, 15)),
        ("6.0.15+detectron2.1", (6, 0, 15)),
        ("6.1.0b2", (6, 1, 0)),
        ("6.1.0rc1", (6, 1, 0)),
    ],
)
def test_parse_release(version, expected):
    assert parse_release(version) == expected


def test_parse_release_rejects_garbage():
    with pytest.raises(ValueError):
        parse_release("not-a-version")


@pytest.fixture(autouse=True)
def isolated_state_file(monkeypatch, tmp_path):
    """Never read or write the state file inside the installed package."""
    monkeypatch.setattr(
        telemetry, "ANALYTICS_STATE_FILE_PATH", tmp_path / "optin.json"
    )


def test_analytics_off_by_default(monkeypatch):
    monkeypatch.delenv(telemetry.ANALYTICS_ENVIRON, raising=False)
    monkeypatch.delenv(telemetry.ANALYTICS_ENABLE_ENVIRON, raising=False)
    assert telemetry.get_usage_analytics_state() is False
    with mock.patch.object(telemetry, "post") as post:
        telemetry.report_usage()
    post.assert_not_called()


def test_analytics_opt_in_through_the_app_switch(monkeypatch):
    monkeypatch.delenv(telemetry.ANALYTICS_ENVIRON, raising=False)
    monkeypatch.delenv(telemetry.ANALYTICS_ENABLE_ENVIRON, raising=False)
    telemetry.set_usage_analytics_state(True)
    assert telemetry.get_usage_analytics_state() is True
    telemetry.set_usage_analytics_state(False)
    assert telemetry.get_usage_analytics_state() is False


def test_analytics_opt_in_sends_only_command_name(monkeypatch):
    monkeypatch.delenv(telemetry.ANALYTICS_ENVIRON, raising=False)
    monkeypatch.setenv(telemetry.ANALYTICS_ENABLE_ENVIRON, "1")
    monkeypatch.setattr(
        telemetry.sys, "argv", ["/bin/idtrackerai", "--video", "/data/v.mp4"]
    )
    with mock.patch.object(telemetry, "post") as post:
        post.return_value.status_code = 200
        telemetry.report_usage()
    payload = post.call_args.kwargs["json"]
    assert payload["command"] == "idtrackerai"
    assert "v.mp4" not in str(payload)


def test_disable_wins_over_enable(monkeypatch):
    monkeypatch.setenv(telemetry.ANALYTICS_ENVIRON, "1")
    monkeypatch.setenv(telemetry.ANALYTICS_ENABLE_ENVIRON, "1")
    assert telemetry.get_usage_analytics_state() is False


def _pypi(*versions):
    html = "\n".join(
        f'<a href="x">idtrackerai-{v}.tar.gz</a><a href="y">idtrackerai-{v}-py3-none-any.whl</a>'
        for v in versions
    )
    return mock.Mock(read=lambda: html.encode())


@pytest.mark.parametrize(
    "current, expected",
    [
        ("6.0.15a0+detectron2.1", ComparisonResult.STABLE_RELEASE),
        ("6.0.15+detectron2.1", ComparisonResult.EQUAL),
        ("6.0.14+detectron2.1", ComparisonResult.PATCH_UPDATE),
        ("5.9.0", ComparisonResult.MAJOR_UPDATE),
        ("6.0.15rc1", ComparisonResult.STABLE_RELEASE),
    ],
)
def test_check_version_local_versions(current, expected):
    telemetry.check_version.cache_clear()
    with (
        mock.patch.object(telemetry, "urlopen", return_value=_pypi("6.0.14", "6.0.15")),
        mock.patch.object(telemetry, "idtrackerai_version", return_value=current),
    ):
        kind, message = telemetry.check_version()
    telemetry.check_version.cache_clear()
    assert kind == expected
    if kind != ComparisonResult.EQUAL:
        assert "mbel-tech/idtrackerai-CYSM" in message
        assert "pip install --upgrade idtrackerai" not in message


def test_check_version_network_error():
    telemetry.check_version.cache_clear()
    with mock.patch.object(telemetry, "urlopen", side_effect=OSError):
        kind, _ = telemetry.check_version()
    telemetry.check_version.cache_clear()
    assert kind == ComparisonResult.ERROR
