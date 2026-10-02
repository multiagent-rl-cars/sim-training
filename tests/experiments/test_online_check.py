"""No-network authentication preflight tests; no real credential access."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from rc_car.experiments import online_check
from rc_car.experiments.config import load_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("NETRC", str(tmp_path / "absent-netrc"))
    fake = MagicMock(
        return_value=SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": {"viewer": {"id": "private-viewer"}, "project": {"name": "sim-training"}}
            },
        )
    )
    monkeypatch.setattr(requests, "post", fake)
    return fake


def recipe():
    return load_config(ROOT / "tests/fixtures/legacy_recipes/line_following_15min.json")


def test_missing_credential_fails_without_network(transport: MagicMock) -> None:
    with pytest.raises(RuntimeError, match="credentials missing"):
        online_check.check_online(recipe())
    transport.assert_not_called()


def test_credential_is_only_sent_to_expected_transport(
    transport: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-secret")
    result = online_check.check_online(recipe())
    assert result["online_verified"] is True
    assert result["run_write_verified"] is False
    assert "synthetic-secret" not in json.dumps(result)
    assert "private-viewer" not in json.dumps(result)
    args, kwargs = transport.call_args
    assert args == ("https://api.wandb.ai/graphql",)
    assert kwargs["auth"] == ("api", "synthetic-secret")
    assert kwargs["timeout"] == (5, 15)
    assert kwargs["allow_redirects"] is False
    assert kwargs["json"]["variables"] == {
        "entity": "2710-multiagent-rl",
        "project": "sim-training",
    }


def test_offline_never_reads_credentials_or_calls_network(
    transport: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(online_check, "configured_key", MagicMock(side_effect=AssertionError))
    assert online_check.check_online(recipe())["status"] == "explicit-offline"
    transport.assert_not_called()


def test_netrc_standard_source(
    transport: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "netrc"
    path.write_text("machine api.wandb.ai login api password synthetic-netrc-secret\n")
    monkeypatch.setenv("NETRC", str(path))
    assert online_check.check_online(recipe())["online_verified"]
    assert transport.call_args.kwargs["auth"] == ("api", "synthetic-netrc-secret")


def test_rejects_nonstandard_host_before_loading_key(
    transport: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_BASE_URL", "https://different.example")
    with pytest.raises(RuntimeError, match="only https://api.wandb.ai"):
        online_check.check_online(recipe())
    transport.assert_not_called()


@pytest.mark.parametrize("status", [301, 401, 403, 500])
def test_http_errors_are_sanitized(
    transport: MagicMock, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-secret")
    transport.return_value.status_code = status
    with pytest.raises(RuntimeError, match="authentication/project check failed") as error:
        online_check.check_online(recipe())
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"errors": ["private server message"]},
        {"data": {"viewer": None, "project": {"name": "sim-training"}}},
        {"data": {"viewer": {"id": "a"}, "project": None}},
        {"data": {"viewer": {"id": "a"}, "project": {"name": "other"}}},
    ],
)
def test_inaccessible_destination_fails(
    transport: MagicMock, monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-secret")
    transport.return_value.json = lambda: payload
    with pytest.raises(RuntimeError, match="destination is not readable"):
        online_check.check_online(recipe())


def test_transport_exception_does_not_expose_request(
    transport: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-secret")
    transport.side_effect = requests.ConnectionError("synthetic-secret")
    with pytest.raises(RuntimeError, match="unreachable") as error:
        online_check.check_online(recipe())
    assert "synthetic-secret" not in str(error.value)
