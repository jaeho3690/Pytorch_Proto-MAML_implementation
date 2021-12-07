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
        self.n_way = meta_config["N"]
        self.k_shot = meta_config["K"]
        self.q_query = meta_config["Q"]
        self.total_episode_num = meta_config["total_episode_num"]
        self.transforms = None
        self.episode_file = (
            f"{meta_config['dataset']}-{self.n_way}way-{self.k_shot}shot-{self.q_query}query-{self.total_episode_num}episodes.pkl"
        )

        # save episodes to reproduce
        if not os.path.exists("episodes"):
            os.makedirs("episodes")

    def _load_episode(self):
        """Load existing episodes"""
        with open("episodes/" + self.episode_file, "rb") as episode_pickle:
            self.episodes = pickle.load(episode_pickle)

    def __len__(self):
        return self.total_episode_num

    def __getitem__(self, index):
        episode = self.episodes[index]
        support_img = self.imagedata[episode["support_idx"], :, :, :]
        support_label = episode["support_label_idx"]
        query_img = self.imagedata[episode["query_idx"], :, :, :]
        query_label = episode["query_label_idx"]
        label_name = episode["label_name"]

        if self.transforms:
            support_img = self.transforms(support_img)
            query_img = self.transforms(query_img)

        return {
            "support_img": support_img,
            "support_label": support_label,
            "query_img": query_img,
            "query_label": query_label,
            "label_name": label_name,
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
        print(f"LOADED EPISODE FILE : {self.episode_file}")

    def _construct_episode(self):
        """Construct dict of episodes if it doesn't exists"""
        episodes = OrderedDict()  # {label_id: {label_id(str),support_idx, query_idx},label_name}
        for i in tqdm(range(self.total_episode_num)):
            support_idx = []
            query_idx = []
            support_label_idx = []
            query_label_idx = []
            # sample from class
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
