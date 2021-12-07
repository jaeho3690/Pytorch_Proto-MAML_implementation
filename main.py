import os
import datetime
import argparse

from utils.utils import NEPTUNE_API_TOKEN, set_seed
import neptune.new as neptune


def main():
    # Load generic meta-learning parameters through argparser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="proto", help="select model type")
    parser.add_argument("--dataset", default="miniimagenet", help="select data type")
    parser.add_argument("--N", default=5, type=int, help="N way")
    parser.add_argument("--K", default=1, type=int, help="K shot")
    parser.add_argument("--Q", default=15, type=int, help="Num of query per class")
    parser.add_argument("--total_episode_num", default=60000, type=int, help="num episodes the model is going to see in total")
    parser.add_argument("--train_iter", default=5000, type=int, help="num of iters in training")
    parser.add_argument("--val_iter", default=5000, type=int, help="num of iters in validation")
    parser.add_argument("--test_iter", default=5000, type=int, help="num of iters in testing")
    parser.add_argument("--load_ckpt", default=None, help="load ckpt")
    parser.add_argument("--save_ckpt", default=None, help="save ckpt")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    meta_config = parser.parse_args()

    run = neptune.init(project="jaeho3690/metalearningCW", api_token=NEPTUNE_API_TOKEN)


if __name__ == "__main__":
    main()
