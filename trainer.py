from models.proto import ProtoNet
from models.maml import MAML

from utils.dataloader import get_dataloader


class Trainer:
    def __init__(self, meta_config, train_config, model_config, logger) -> None:
        self.meta_config = meta_config
        self.train_config = train_config
        self.model_config = model_config
        self.logger = logger

        if meta_config["model"] == "proto":
            self.model = ProtoNet()
        elif meta_config["model"] == "maml":
            self.model = MAML()
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(meta_config, train_config)

    def train_step(self):
        pass

    def val_step(self):
        pass

    def test_step(self):
        pass

    def run_experiment(self):
        pass
