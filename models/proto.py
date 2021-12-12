import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class ProtoNet(nn.Module):
    def __init__(self, meta_config, model_config, learner, logger):
        super(ProtoNet, self).__init__()
        self.meta_config = meta_config
        self.model_config = model_config
        self.learner = learner
        self.logger = logger
        self.device = meta_config["device"]
        self.train_iter = meta_config["train_iter"]
        self.val_iter = meta_config["val_iter"]
        self.train_accuracy = []
        self.val_accuracy = []

        self.total_train_episode_num = meta_config["total_train_episode_num"]
        self.train_iter = meta_config["train_iter"]
        self.val_iter = meta_config["val_iter"]
        self.n_way = meta_config["N"]
        self.k_shot = meta_config["K"]

        self.best_val_accuracy = 0

        self.learner.to(device=self.device)

        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=1e-3)

    def calculate_prototypes(self, support_embeddings, support_labels):
        prototypes = []

        support_labels_expand = support_labels.view(-1, 1).expand((support_embeddings.shape))
        # aggregate samples from the same class. average them to get prototype
        for label in range(self.n_way):
            proto = torch.mean(support_embeddings[support_labels_expand == label].view(self.k_shot, -1), dim=0)
            prototypes.append(proto)

        return torch.stack(prototypes)

    def prototypical_loss(self, prototypes, query_embeddings, query_labels):
        distance_matrix = ((query_embeddings[:, :, None] - prototypes.t()[None, :, :]) ** 2).sum(1)
        return F.cross_entropy(-distance_matrix, query_labels.squeeze()), distance_matrix

    def train(self, train_loader, val_loader):

        # loads a single episode
        for episode_idx, episode in enumerate(tqdm(train_loader)):
            self.learner.train()
            self.optimizer.zero_grad()

            support_images = episode["support_img"].to(self.device)
            query_images = episode["query_img"].to(self.device)
            support_labels = episode["support_label"].to(self.device)
            query_labels = episode["query_label"].to(self.device)

            # get embeddings
            support_embeddings = self.learner(support_images.squeeze())
            query_embeddings = self.learner(query_images.squeeze())

            # calculate prototypes and the loss
            prototypes = self.calculate_prototypes(support_embeddings, support_labels)
            loss, distance_matrix = self.prototypical_loss(prototypes, query_embeddings, query_labels)

            # update
            loss.backward()
            self.optimizer.step()

            # Logging
            accuracy = self.calculate_accuracy(distance_matrix, query_labels)
            self.train_accuracy.append(accuracy)
            self.logger["train/episode/accuracy"].log(accuracy)

            # run validation when episode_idx  == self.train_iter
            if (episode_idx % self.train_iter == 0) and (episode_idx != 0):
                self.validate(val_loader)

            if episode_idx == self.total_train_episode_num:
                train_log = pd.DataFrame(self.train_accuracy)
                val_log = pd.DataFrame(self.val_accuracy)
                train_log.to_csv(f"logging/{self.meta_config['save_pt']}-train.csv")
                val_log.to_csv(f"logging/{self.meta_config['save_pt']}-val.csv")

    def validate(self, val_loader):
        print("run validation!")
        with torch.no_grad():
            val_accuracy_lists = []
            for val_episode_idx, episode in enumerate(tqdm(val_loader)):
                support_images = episode["support_img"].to(self.device)
                query_images = episode["query_img"].to(self.device)
                support_labels = episode["support_label"].to(self.device)
                query_labels = episode["query_label"].to(self.device)

                support_embeddings = self.learner(support_images.squeeze())
                query_embeddings = self.learner(query_images.squeeze())

                prototypes = self.calculate_prototypes(support_embeddings, support_labels)
                _, distance_matrix = self.prototypical_loss(prototypes, query_embeddings, query_labels)
                accuracy = self.calculate_accuracy(distance_matrix, query_labels)

                self.val_accuracy.append(accuracy)
                self.logger["val/episode/accuracy"].log(accuracy)
                val_accuracy_lists.append(accuracy)

                # Save when better than previous mean val accuracy
                if val_episode_idx == self.val_iter:
                    if np.mean(val_accuracy_lists) > self.best_val_accuracy:
                        self.best_val_accuracy = np.mean(val_accuracy_lists)
                        filename = f"checkpoints/{self.meta_config['save_pt']}.pt"

                        print(f"\nSaving state dictionary to {filename}")
                        with open(filename, "wb") as f:
                            state_dict = self.learner.state_dict()
                            torch.save(state_dict, f)

                    return

    def test(self, test_loader):
        with torch.no_grad():
            test_accuracy_lists = []
            for test_episode_idx, episode in enumerate(tqdm(test_loader)):
                support_images = episode["support_img"].to(self.device)
                query_images = episode["query_img"].to(self.device)
                support_labels = episode["support_label"].to(self.device)
                query_labels = episode["query_label"].to(self.device)

                support_embeddings = self.learner(support_images.squeeze())
                query_embeddings = self.learner(query_images.squeeze())

                prototypes = self.calculate_prototypes(support_embeddings, support_labels)
                _, distance_matrix = self.prototypical_loss(prototypes, query_embeddings, query_labels)
                accuracy = self.calculate_accuracy(distance_matrix, query_labels)
                test_accuracy_lists.append(accuracy)

            print(f"Test Accuracy at {self.n_way}Way-{self.k_shot}Shot: {np.mean(test_accuracy_lists)}")
            self.logger["test/accuracy"].log(accuracy)

    def calculate_accuracy(self, distance_matrix, query_labels):
        with torch.no_grad():
            pred_label = torch.min(distance_matrix, 1)[1]
            accuracy = torch.sum(pred_label == query_labels) / query_labels.shape[1]
        return accuracy.item()
