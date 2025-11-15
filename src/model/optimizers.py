from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from model.model import ReProSeg, TrainPhase


class OptimizerSchedulerManager:
    def __init__(self, model: ReProSeg, t_max_backbone, lr_backbone):
        self.optimizer_net, self.optimizer_classifier = model.get_optimizers()
        self.scheduler_net = CosineAnnealingLR(
            self.optimizer_net,
            T_max=t_max_backbone,
            eta_min=lr_backbone / 100,
        )
        # scheduler for the classification layer is with restarts,
        # such that the model can re-activated zeroed-out prototypes.
        # Hence, an intuitive choice.
        # self.scheduler_classifier = CosineAnnealingWarmRestarts(
        #     self.optimizer_classifier,
        #     T_0=5,
        #     eta_min=0.001,
        # )

        self.scheduler_classifier = CosineAnnealingLR(
            self.optimizer_classifier,
            T_max=3000,
            eta_min=0.001,
        )

    def load_state_dict(self, state_dict):
        if "optimizer_net_state_dict" in state_dict:
            self.optimizer_net.load_state_dict(state_dict["optimizer_net_state_dict"])
        if "scheduler_net_state_dict" in state_dict:
            self.scheduler_net.load_state_dict(state_dict["scheduler_net_state_dict"])
        if "optimizer_classifier_state_dict" in state_dict:
            self.optimizer_classifier.load_state_dict(state_dict["optimizer_classifier_state_dict"])
        if "scheduler_classifier_state_dict" in state_dict:
            self.scheduler_classifier.load_state_dict(state_dict["scheduler_classifier_state_dict"])

    def reset_gradients(self):
        self.optimizer_net.zero_grad()
        self.optimizer_classifier.zero_grad()

    def step(self, train_phase: TrainPhase, epoch):
        if train_phase is not TrainPhase.PRETRAIN:
            self.optimizer_classifier.step()
            self.scheduler_classifier.step(epoch)

        if train_phase is not TrainPhase.FINETUNE:
            self.optimizer_net.step()
            self.scheduler_net.step()

    def get_checkpoint(self):
        checkpoint = {
            "optimizer_net_state_dict": self.optimizer_net.state_dict(),
            "scheduler_net_state_dict": self.scheduler_net.state_dict(),
        }
        if self.optimizer_classifier is not None:
            checkpoint["optimizer_classifier_state_dict"] = self.optimizer_classifier.state_dict()
        if self.scheduler_classifier is not None:
            checkpoint["scheduler_classifier_state_dict"] = self.scheduler_classifier.state_dict()
        return checkpoint
