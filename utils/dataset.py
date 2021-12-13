import pandas as pd
import numpy as np
import pickle
import os
import random
from random import sample
from collections import OrderedDict

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MetaDataset(Dataset):
    def __init__(self, mode, meta_config):
        """
        Args:
            mode: [train, val, test]
            meta_config: argparser from main.py
        """
        self.mode = mode
        self.meta_config = meta_config
        self.data_dir = "data/" + meta_config["dataset"]
        if self.mode in ("train"):
            self.n_way = meta_config["N"]
            self.k_shot = meta_config["K"]
            self.q_query = meta_config["Q"]
        elif self.mode in ("val", "test"):
            self.n_way = meta_config["valtest_N"]
            self.k_shot = meta_config["valtest_K"]
            self.q_query = meta_config["valtest_Q"]
        self.total_episode_num = meta_config[f"total_{self.mode}_episode_num"]
        self.transform = None
        self.episode_file = f"{meta_config['dataset']}-{self.mode}-{self.n_way}way-{self.k_shot}shot-{self.q_query}query-{self.total_episode_num}episodes.pkl"

        # save episodes to reproduce
        if not os.path.exists("episodes"):
            os.makedirs("episodes")

    def _load_episode(self):
        """Load existing episodes"""
        with open("episodes/" + self.episode_file, "rb") as episode_pickle:
            self.episodes = pickle.load(episode_pickle)

    def __len__(self):
        return self.meta_config[f"total_{self.mode}_episode_num"]

    def __getitem__(self, index):
        # load episode.
        episode = self.episodes[index]
        support_img = self.imagedata[episode["support_idx"], :, :, :]
        support_label = torch.tensor(episode["support_label_idx"], dtype=torch.long)
        query_img = self.imagedata[episode["query_idx"], :, :, :]
        query_label = torch.tensor(episode["query_label_idx"], dtype=torch.long)
        # label_name = episode['label_name']

        if self.transform:
            # apply transform per image and stack them back
            # transform converts (n_rows, n_cols, n_channels) to (n_channels, n_rows, n_cols).
            # ToTensor converts np array into torch tensor with value between 0 and 1
            support_img = torch.stack([self.transform(support_img[b, :, :, :]) for b in range(support_img.shape[0])])
            query_img = torch.stack([self.transform(query_img[b, :, :, :]) for b in range(query_img.shape[0])])

        return {
            "support_img": support_img,
            "support_label": support_label,
            "query_img": query_img,
            "query_label": query_label,
            # "label_name": label_name,
        }


class MiniImagenetDataset(MetaDataset):
    """Initialize the dataset with the total number of episodes for training. The loader will take care of the rest"""

    def __init__(self, **kwargs):
        super(MiniImagenetDataset, self).__init__(**kwargs)
        with open(self.data_dir + "/mini-imagenet-classlabel.pkl", "rb") as file:
            self.id2label = pickle.load(file)

        if self.mode == "train":
            dataset = pd.read_pickle(self.data_dir + "/mini-imagenet-cache-train.pkl")
        elif self.mode == "val":
            dataset = pd.read_pickle(self.data_dir + "/mini-imagenet-cache-val.pkl")
        elif self.mode == "test":
            dataset = pd.read_pickle(self.data_dir + "/mini-imagenet-cache-test.pkl")

        self.imagedata = dataset["image_data"]
        self.label_dict = dataset["class_dict"]
        self.unique_labels = list(self.label_dict.keys())
        self.transforms = None

        if os.path.isfile("episodes/" + self.episode_file):
            self._load_episode()
        else:
            self._construct_episode()
        print(f"LOADED EPISODE FILE FOR {self.mode}: {self.episode_file}")

        # TODO: Implement
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

    def _construct_episode(self):
        """Construct dict of episodes if it doesn't exists"""
        episodes = OrderedDict()  # {label_id: {label_id(str),support_idx, query_idx},label_name}
        print(f"Building episodes from scratch")

        # TODO: change self.total_epsiode_num

        for i in tqdm(range(self.meta_config[f"total_{self.mode}_episode_num"])):
            support_idx = []
            query_idx = []
            support_label_idx = []
            query_label_idx = []
            # sample from class
            assert (
                len(self.unique_labels) >= self.n_way
            ), f"Number of unique labels({len(self.unique_labels)}) should be bigger than the number of ways({self.n_way})"
            label_ids = sample(self.unique_labels, self.n_way)
            label_name = [self.id2label[id] for id in label_ids]

            # sample image idx from selected classes
            for idx, label in enumerate(label_ids):
                sampled_label_idxs = sample(self.label_dict[label], self.k_shot + self.q_query)
                support_idx.extend(sampled_label_idxs[: self.k_shot])
                query_idx.extend(sampled_label_idxs[self.k_shot :])
                support_label_idx.extend([idx for _ in range(self.k_shot)])
                query_label_idx.extend([idx for _ in range(self.q_query)])

            # shuffle in the same order. Make reordering index
            support_shuffled_index = sample([i for i in range(len(support_idx))], len(support_idx))
            query_shuffled_index = sample([i for i in range(len(query_idx))], len(query_idx))
            # shuffle
            support_idx = [support_idx[i] for i in support_shuffled_index]
            support_label_idx = [support_label_idx[i] for i in support_shuffled_index]
            query_idx = [query_idx[i] for i in query_shuffled_index]
            query_label_idx = [query_label_idx[i] for i in query_shuffled_index]

            # make episode
            episodes[i] = {
                "label_id": label_ids,
                "label_name": label_name,
                "support_idx": support_idx,
                "query_idx": query_idx,
                "support_label_idx": support_label_idx,
                "query_label_idx": query_label_idx,
            }

        self.episodes = episodes
        # save to pickle file
        with open("episodes/" + self.episode_file, "wb") as episode_pickle:
            pickle.dump(episodes, episode_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # For debugging
    meta_train = MiniImagenetDataset(
        mode="train", meta_config={"dataset": "miniimagenet", "N": 5, "K": 1, "Q": 1, "total_episode_num": 1000, "seed": 1}
    )
