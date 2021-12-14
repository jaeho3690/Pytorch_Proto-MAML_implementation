"""I took reference from https://github.com/dragen1860/MAML-Pytorch """
import pandas as pd
import numpy as np
import scipy.stats as st
import copy

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class MAML(nn.Module):
    def __init__(self, meta_config, model_config, learner, logger):
        super(MAML, self).__init__()
        self.meta_config = meta_config
        self.model_config = model_config
        self.learner = learner
        self.device = meta_config["device"]
        self.logger = logger

        self.total_train_episode_num = meta_config["total_train_episode_num"]
        self.train_iter = meta_config["train_iter"]
        self.val_iter = meta_config["val_iter"]
        self.test_iter = meta_config["test_iter"]
        self.train_accuracy = []
        self.val_accuracy = []

        self.num_tasks_for_inner_update = model_config["inner_loop"]
        self.num_inner_updates = model_config["inner_update_steps"]
        self.num_inner_updates_test = model_config["inner_update_steps_test"]
        self.inner_lr = model_config["inner_lr"]
        self.outer_lr = model_config["outer_lr"]
        self.n_way = meta_config["N"]
        self.k_shot = meta_config["K"]
        self.best_val_accuracy = 0

        self.learner.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=self.outer_lr)

    def train(self, train_loader, val_loader):
        """Receives the data_loader and run inner update"""

        # Calculate when to run the validation loop.
        check_val_idx = self.train_iter // self.num_tasks_for_inner_update

        # This loads `num_tasks_for_inner_update` number of episodes per single loop
        # Here we load 4 episodes at once.
        for outer_update_idx, episodes in enumerate(tqdm(train_loader)):
            support_images = episodes["support_img"].to(self.device)
            query_images = episodes["query_img"].to(self.device)
            support_labels = episodes["support_label"].to(self.device)
            query_labels = episodes["query_label"].to(self.device)

            inner_losses = 0

            # Loop around the inner tasks (4 episodes, other number of episode also works fine)
            for inner_episode_idx in range(self.num_tasks_for_inner_update):
                # First update of initial parameter on support set
                support_logit = self.learner(support_images[inner_episode_idx, :, :, :, :], None, bn_training=True)
                support_loss = F.cross_entropy(support_logit, support_labels[inner_episode_idx])
                grad = torch.autograd.grad(support_loss, self.learner.parameters())
                theta_prime = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, self.learner.parameters())))

                # if you want more than a single update. Beware that range starts from 1.
                for _ in range(1, self.num_inner_updates):
                    support_logit = self.learner(support_images[inner_episode_idx, :, :, :, :], theta_prime, bn_training=True)
                    support_loss = F.cross_entropy(support_logit, support_labels[inner_episode_idx])
                    grad = torch.autograd.grad(support_loss, theta_prime)
                    theta_prime = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, theta_prime)))

                # Calculate loss on query update.
                query_logit = self.learner(query_images[inner_episode_idx, :, :, :, :], theta_prime, bn_training=True)
                query_loss = F.cross_entropy(query_logit, query_labels[inner_episode_idx])

                # add the loss
                inner_losses += query_loss

                with torch.no_grad():
                    accuracy = self.calculate_accuracy(query_logit, query_labels[inner_episode_idx])
                    self.train_accuracy.append(accuracy)
                    self.logger["train/episode/accuracy"].log(accuracy)

            # divide the loss based on the number of inner episodes
            inner_losses = inner_losses / self.num_tasks_for_inner_update
            self.optimizer.zero_grad()
            inner_losses.backward()
            self.optimizer.step()

            # Log train and validation results
            if (self.total_train_episode_num // self.num_tasks_for_inner_update) == outer_update_idx + 1:
                train_log = pd.DataFrame(self.train_accuracy)
                val_log = pd.DataFrame(self.val_accuracy)
                train_log.to_csv(f"logging/{self.meta_config['save_pt']}-train.csv")
                val_log.to_csv(f"logging/{self.meta_config['save_pt']}-val.csv")
                self.logger["train/preds"].upload(f"logging/{self.meta_config['save_pt']}-train.csv")
                return

            # Run validation every check_val_idx
            if (outer_update_idx % check_val_idx == 0) and (outer_update_idx != 0):
                self.validate(val_loader)

    def calculate_accuracy(self, logit, label):
        with torch.no_grad():
            pred = torch.max(logit, 1)[1]
            accuracy = torch.sum(pred == label) / label.shape[0]
        return accuracy.item()

    def validate(self, val_loader):
        print("run validation!")
        val_accuracy_lists = []
        stop_idx = self.val_iter // self.num_tasks_for_inner_update

        # Copy the learner so gradient is not affected
        validation_learner = copy.deepcopy(self.learner)

        for outer_update_idx, episodes in enumerate(tqdm(val_loader)):
            support_images = episodes["support_img"].to(self.device)
            query_images = episodes["query_img"].to(self.device)
            support_labels = episodes["support_label"].to(self.device)
            query_labels = episodes["query_label"].to(self.device)

            for inner_episode_idx in range(self.num_tasks_for_inner_update):
                support_logit = validation_learner(support_images[inner_episode_idx, :, :, :].squeeze(), None, bn_training=False)
                support_loss = F.cross_entropy(support_logit, support_labels[inner_episode_idx])
                grad = torch.autograd.grad(support_loss, validation_learner.parameters())
                theta_prime = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, validation_learner.parameters())))

                # if you want several updates
                for _ in range(1, self.num_inner_updates):
                    support_logit = validation_learner(
                        support_images[inner_episode_idx, :, :, :, :], theta_prime, bn_training=True
                    )
                    support_loss = F.cross_entropy(support_logit, support_labels[inner_episode_idx])
                    grad = torch.autograd.grad(support_loss, theta_prime)
                    theta_prime = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, theta_prime)))

                query_logit = validation_learner(query_images[inner_episode_idx, :, :, :, :], theta_prime, bn_training=True)

                with torch.no_grad():
                    accuracy = self.calculate_accuracy(query_logit, query_labels[inner_episode_idx])
                    self.val_accuracy.append(accuracy)
                    val_accuracy_lists.append(accuracy)
                    self.logger["val/episode/accuracy"].log(accuracy)

            if outer_update_idx == stop_idx:
                if np.mean(val_accuracy_lists) > self.best_val_accuracy:
                    self.best_val_accuracy = np.mean(val_accuracy_lists)
                    filename = f"checkpoints/{self.meta_config['save_pt']}.pt"

                    print(f"\nSaving state dictionary to {filename}")
                    with open(filename, "wb") as f:
                        state_dict = self.learner.state_dict()
                        torch.save(state_dict, f)
                return

    def test(self, test_loader):
        print("run test!")
        test_accuracy_lists = []
        # Copy the learner so gradient is not affected
        test_learner = copy.deepcopy(self.learner)

        for outer_update_idx, episodes in enumerate(tqdm(test_loader)):
            support_images = episodes["support_img"].to(self.device)
            query_images = episodes["query_img"].to(self.device)
            support_labels = episodes["support_label"].to(self.device)
            query_labels = episodes["query_label"].to(self.device)

            for inner_episode_idx in range(self.num_tasks_for_inner_update):
                support_logit = test_learner(support_images[inner_episode_idx, :, :, :].squeeze(), None, bn_training=False)
                support_loss = F.cross_entropy(support_logit, support_labels[inner_episode_idx])
                grad = torch.autograd.grad(support_loss, test_learner.parameters())
                theta_prime = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, test_learner.parameters())))

                # if you want several updates
                for _ in range(1, self.num_inner_updates_test):
                    support_logit = test_learner(support_images[inner_episode_idx, :, :, :, :], theta_prime, bn_training=True)
                    support_loss = F.cross_entropy(support_logit, support_labels[inner_episode_idx])
                    grad = torch.autograd.grad(support_loss, theta_prime)
                    theta_prime = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, theta_prime)))

                query_logit = test_learner(query_images[inner_episode_idx, :, :, :, :], theta_prime, bn_training=True)

                with torch.no_grad():
                    accuracy = self.calculate_accuracy(query_logit, query_labels[inner_episode_idx])
                    test_accuracy_lists.append(accuracy)
                    self.logger["test/episode/accuracy"].log(accuracy)

        # Log test
        test_log = pd.DataFrame(test_accuracy_lists, columns=["Accuracy"])
        test_log.to_csv(f"logging/{self.meta_config['save_pt']}-test.csv")
        # Save 95% interval on test accuracy
        (low, high) = st.t.interval(
            alpha=0.95, df=len(test_accuracy_lists) - 1, loc=np.mean(test_accuracy_lists), scale=st.sem(test_accuracy_lists)
        )
        print(f"Test Accuracy at {self.n_way}Way-{self.k_shot}Shot: {np.mean(test_accuracy_lists)}")
        print(f"Test Accuracy 95% interval :{low:.3f},{high:.3f}")
        self.logger["test/preds"].upload(f"logging/{self.meta_config['save_pt']}-test.csv")
        self.logger["test/accuracy"].log(np.mean(test_accuracy_lists))
        self.logger["test/confidence"].log(high - accuracy)
        self.logger["test/low_95accuracy"].log(low)
        self.logger["test/high_95accuracy"].log(high)
