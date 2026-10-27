from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config.schema.model import LossCriterion
from train import trainer
from train.test_step import EvalInfo
from utils.run_context import RunContext


def test_train_model_logs_evaluation_sparsity_and_abstained(tmp_path, monkeypatch, mock_config):
    import utils.run_context as run_context

    cfg = mock_config
    cfg.env.device = torch.device("cpu")
    cfg.model.criterion = LossCriterion.NLL
    cfg.training.epochs.total = 1
    xs = torch.zeros((1, 2, 1, 1))
    loader = [(xs, torch.zeros((1, 1, 1, 1), dtype=torch.long))]
    for name in ("AugmentedDataset", "Dataset"):
        monkeypatch.setattr(trainer, name, MagicMock())
    monkeypatch.setattr(trainer, "DataLoader", MagicMock(return_value=loader))
    manager = MagicMock()
    manager.get_checkpoint.return_value = {}
    monkeypatch.setattr(trainer, "OptimizerSchedulerManager", MagicMock(return_value=manager))
    monkeypatch.setattr(trainer, "compat_version", MagicMock(return_value="0.2"))
    monkeypatch.setattr(trainer, "get_class_weights", MagicMock(return_value=torch.ones(2)))
    train_info = SimpleNamespace(
        accuracy=0.0,
        miou=0.0,
        loss=SimpleNamespace(**{name: torch.tensor(0.0) for name in ("total", "jsd", "tanh", "classification")}),
    )
    monkeypatch.setattr(trainer, "train", MagicMock(return_value=train_info))
    eval_info = EvalInfo(
        abstained=0.25,
        sparsity=0.5,
        num_non_zero_concepts=2,
        confusion_matrix=np.zeros((1, 1)),
        accuracy=0.0,
        miou=0.0,
    )
    evaluate = MagicMock(return_value=eval_info)
    monkeypatch.setattr(trainer, "eval", evaluate)
    monkeypatch.setattr(trainer.nni, "report_intermediate_result", MagicMock())
    monkeypatch.setattr(trainer.nni, "report_final_result", MagicMock())
    net = MagicMock(return_value=(xs, xs, xs))
    run = RunContext(tmp_path)
    monkeypatch.setattr(run_context, "_run_context", run)
    monkeypatch.setattr(run, "model_checkpoint", MagicMock())
    try:
        trainer.train_model(net, [], [], cfg)
    finally:
        run.close()

    evaluate.assert_called_once_with(cfg, net, loader, 1)
    events = EventAccumulator(str(run.tensorboard_dir)).Reload()
    for tag, expected in [
        ("sparsity/eval-epochs", eval_info.sparsity),
        ("abstained/eval-epochs", eval_info.abstained),
    ]:
        (scalar,) = events.Scalars(tag)
        assert scalar.step == 1
        assert scalar.value == pytest.approx(expected)
