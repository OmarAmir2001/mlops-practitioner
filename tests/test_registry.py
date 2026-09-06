from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import RestException

from prodml import registry


def _fake_run(metrics):
    run = MagicMock()
    run.data.metrics = metrics
    return run


def _fake_version(version, run_id):
    v = MagicMock()
    v.version = version
    v.run_id = run_id
    return v


def test_find_version_for_run(monkeypatch):
    monkeypatch.setattr(
        registry.client,
        "search_model_versions",
        lambda q: [_fake_version("1", "aaa"), _fake_version("2", "bbb")],
    )
    assert registry._find_version_for_run("churn-predictor", "bbb") == "2"


def test_find_version_raises_when_absent(monkeypatch):
    monkeypatch.setattr(registry.client, "search_model_versions", lambda q: [])
    with pytest.raises(ValueError):
        registry._find_version_for_run("churn-predictor", "nope")


def test_promotes_when_nothing_in_production(monkeypatch):
    monkeypatch.setattr(
        registry.client, "get_run", lambda rid: _fake_run({"roc_auc": 0.90})
    )
    monkeypatch.setattr(
        registry.client,
        "search_model_versions",
        lambda q: [_fake_version("1", "new")],
    )

    def raise_not_found(name, alias):
        raise RestException({"error_code": "RESOURCE_DOES_NOT_EXIST"})

    monkeypatch.setattr(registry.client, "get_model_version_by_alias", raise_not_found)
    set_alias = MagicMock()
    monkeypatch.setattr(registry.client, "set_registered_model_alias", set_alias)

    assert registry.promote_if_better("new") is True
    set_alias.assert_called_once()


def test_promotes_when_better(monkeypatch):
    runs = {"new": _fake_run({"roc_auc": 0.90}), "old": _fake_run({"roc_auc": 0.80})}
    monkeypatch.setattr(registry.client, "get_run", lambda rid: runs[rid])
    monkeypatch.setattr(
        registry.client,
        "search_model_versions",
        lambda q: [_fake_version("2", "new")],
    )
    monkeypatch.setattr(
        registry.client,
        "get_model_version_by_alias",
        lambda name, alias: _fake_version("1", "old"),
    )
    set_alias = MagicMock()
    monkeypatch.setattr(registry.client, "set_registered_model_alias", set_alias)

    assert registry.promote_if_better("new") is True
    set_alias.assert_called_once_with("churn-predictor", registry.PRODUCTION_ALIAS, "2")


def test_declines_when_worse(monkeypatch):
    runs = {"new": _fake_run({"roc_auc": 0.70}), "old": _fake_run({"roc_auc": 0.80})}
    monkeypatch.setattr(registry.client, "get_run", lambda rid: runs[rid])
    monkeypatch.setattr(
        registry.client,
        "search_model_versions",
        lambda q: [_fake_version("2", "new")],
    )
    monkeypatch.setattr(
        registry.client,
        "get_model_version_by_alias",
        lambda name, alias: _fake_version("1", "old"),
    )
    set_alias = MagicMock()
    monkeypatch.setattr(registry.client, "set_registered_model_alias", set_alias)

    assert registry.promote_if_better("new") is False
    set_alias.assert_not_called()


def test_declines_when_equal(monkeypatch):
    """A tie is not an improvement."""
    runs = {"new": _fake_run({"roc_auc": 0.80}), "old": _fake_run({"roc_auc": 0.80})}
    monkeypatch.setattr(registry.client, "get_run", lambda rid: runs[rid])
    monkeypatch.setattr(
        registry.client,
        "search_model_versions",
        lambda q: [_fake_version("2", "new")],
    )
    monkeypatch.setattr(
        registry.client,
        "get_model_version_by_alias",
        lambda name, alias: _fake_version("1", "old"),
    )
    monkeypatch.setattr(registry.client, "set_registered_model_alias", MagicMock())

    assert registry.promote_if_better("new") is False
