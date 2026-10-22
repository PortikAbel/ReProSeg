from config.schema.training import SchedulersConfig
from model.model import ReProSeg, TrainPhase
from train.optim.schedulers import build_scheduler


class OptimizerSchedulerManager:
    def __init__(
        self,
        model: ReProSeg,
        schedulers_cfg: SchedulersConfig,
        iters_per_epoch: int,
        epochs: int,
    ):
        self.scheduler_types = {
            "net": schedulers_cfg.net.type,
            "classifier": schedulers_cfg.classifier.type,
        }
        self.optimizer_net, self.optimizer_classifier = model.get_optimizers()
        max_iterations = iters_per_epoch * epochs
        self.scheduler_net = build_scheduler(self.optimizer_net, schedulers_cfg.net, iters_per_epoch, max_iterations)
        self.scheduler_classifier = build_scheduler(
            self.optimizer_classifier, schedulers_cfg.classifier, iters_per_epoch, max_iterations
        )

    def load_state_dict(self, state_dict):
        if "optimizer_net_state_dict" in state_dict:
            self.optimizer_net.load_state_dict(state_dict["optimizer_net_state_dict"])
        for name, scheduler_type in self.scheduler_types.items():
            saved_type = state_dict.get(f"scheduler_{name}_type")
            if saved_type is not None and saved_type != scheduler_type:
                raise ValueError(
                    f"checkpoint {name} scheduler is '{saved_type}' but config requests '{scheduler_type}'"
                )
        if "scheduler_net_state_dict" in state_dict:
            self.scheduler_net.load_state_dict(state_dict["scheduler_net_state_dict"])
        if "optimizer_classifier_state_dict" in state_dict:
            self.optimizer_classifier.load_state_dict(state_dict["optimizer_classifier_state_dict"])
        if "scheduler_classifier_state_dict" in state_dict:
            self.scheduler_classifier.load_state_dict(state_dict["scheduler_classifier_state_dict"])

    def reset_gradients(self):
        self.optimizer_net.zero_grad()
        self.optimizer_classifier.zero_grad()

    def step(self, train_phase: TrainPhase):
        self.optimizer_classifier.step()
        self.scheduler_classifier.step()

        if train_phase is not TrainPhase.FINETUNE:
            self.optimizer_net.step()
            self.scheduler_net.step()

    def get_checkpoint(self):
        checkpoint = {
            "optimizer_net_state_dict": self.optimizer_net.state_dict(),
            "scheduler_net_state_dict": self.scheduler_net.state_dict(),
            "scheduler_net_type": self.scheduler_types["net"],
            "scheduler_classifier_type": self.scheduler_types["classifier"],
        }
        if self.optimizer_classifier is not None:
            checkpoint["optimizer_classifier_state_dict"] = self.optimizer_classifier.state_dict()
        if self.scheduler_classifier is not None:
            checkpoint["scheduler_classifier_state_dict"] = self.scheduler_classifier.state_dict()
        return checkpoint
