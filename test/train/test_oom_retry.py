from types import SimpleNamespace

import pytest
import torch

from train import __main__ as entry


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(entry, "ReProSeg", lambda cfg: torch.nn.Linear(1, 1))
    return SimpleNamespace(
        training=SimpleNamespace(oom_retry=True),
        data=SimpleNamespace(batch_size=16),
        env=SimpleNamespace(device=torch.device("cpu")),
    )


def _fake_train(max_ok, seen):
    def train_model(net, train, valid, cfg):
        seen.append(cfg.data.batch_size)
        if cfg.data.batch_size > max_ok:
            raise torch.cuda.OutOfMemoryError("oom")

    return train_model


def test_halves_until_fits(cfg, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(entry, "train_model", _fake_train(4, seen))
    entry.train_with_oom_retry(None, None, cfg)
    assert seen == [16, 8, 4]


def test_reraises_when_disabled(cfg, monkeypatch):
    cfg.training.oom_retry = False
    seen: list[int] = []
    monkeypatch.setattr(entry, "train_model", _fake_train(4, seen))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        entry.train_with_oom_retry(None, None, cfg)
    assert seen == [16]


def test_reraises_at_floor(cfg, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(entry, "train_model", _fake_train(1, seen))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        entry.train_with_oom_retry(None, None, cfg)
    assert seen == [16, 8, 4, 2]
