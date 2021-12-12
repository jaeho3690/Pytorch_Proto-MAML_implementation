from models.proto import ProtoNet
from models.maml import MAML

from utils.dataloader import get_dataloader
from models.cnn import Learner
from models.cnn import ProtoLearner


class Trainer:
    def __init__(self, meta_config, learner_config, model_config, logger) -> None:
        self.meta_config = meta_config
        self.model_config = model_config
        self.logger = logger
        self.device = meta_config["device"]

        if meta_config["model"] == "proto":
            learner = ProtoLearner(model_config["in_channels"], model_config["out_channels"])
            self.model = ProtoNet(meta_config=meta_config, model_config=model_config, learner=learner, logger=logger).to(
                self.device
            )
        elif meta_config["model"] == "maml":
            learner = Learner(learner_config)
            self.model = MAML(meta_config=meta_config, model_config=model_config, learner=learner, logger=logger).to(self.device)
        else:
            print("ADD YOUR MODEL HERE")

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(meta_config, model_config)

    def run_experiment(self):

        self.model.train(train_loader=self.train_loader, val_loader=self.val_loader)
        self.model.test(test_loader=self.test_loader)

    def evaluate(self, test_loader):
        # self.model
        pass

    def load_checkpoint(self, checkpoint):
        pass
