import os
import datetime
import argparse
import yaml

from utils.utils import NEPTUNE_API_TOKEN, set_seed
from trainer import Trainer
import neptune.new as neptune


def main():
    # Load generic meta-learning parameters through argparser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="maml", help="select model type['maml','proto']")
    parser.add_argument("--dataset", default="miniimagenet", help="select data type")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--N", default=5, type=int, help="train episode N way")
    parser.add_argument("--K", default=1, type=int, help="train episode K shot")
    parser.add_argument("--Q", default=15, type=int, help="train episode num of query per class")
    parser.add_argument("--valtest_N", default=5, type=int, help="Val/Test episode N way")
    parser.add_argument("--valtest_K", default=1, type=int, help="Val/Test episode K shot")
    parser.add_argument("--valtest_Q", default=15, type=int, help="Val/Test episode num of query per class")
    parser.add_argument("--total_train_episode_num", default=60000, type=int, help="num episodes in train")
    parser.add_argument("--total_val_episode_num", default=60000, type=int, help="num episodes in val")
    parser.add_argument("--total_test_episode_num", default=600, type=int, help="num episodes in test")
    parser.add_argument("--train_iter", default=5000, type=int, help="num of iters in training")
    parser.add_argument("--val_iter", default=5000, type=int, help="num of iters in validation")
    parser.add_argument("--test_iter", default=600, type=int, help="num of iters in testing")
    parser.add_argument("--load_pt", default=None, help="load pt")
    parser.add_argument("--save_pt", default=None, help="save pt")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    meta_config = vars(parser.parse_args())

    meta_config[
        "save_pt"
    ] = f"{meta_config['model']}-{meta_config['dataset']}-{meta_config['N']}way-{meta_config['K']}shot-{meta_config['Q']}-testN{meta_config['valtest_N']}K{meta_config['valtest_K']}"

    with open(f"configs/{meta_config['model']}_configs.yaml") as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
        model_config = configs[0]["model_config"]
    # logger = None
    logger = neptune.init(project="jaeho3690/metalearningCW", api_token=NEPTUNE_API_TOKEN)
    logger["sys/tags"].add(
        [meta_config["model"], meta_config["dataset"], str(meta_config["N"]), str(meta_config["K"]), str(meta_config["Q"])]
    )
    logger["meta_config"] = meta_config
    logger["model_config"] = model_config

    # logger = None
    learner_config = [
        ("conv2d", [32, 3, 3, 3, 1, 0]),
        ("relu", [True]),
        ("bn", [32]),
        ("max_pool2d", [2, 2, 0]),
        ("conv2d", [32, 32, 3, 3, 1, 0]),
        ("relu", [True]),
        ("bn", [32]),
        ("max_pool2d", [2, 2, 0]),
        ("conv2d", [32, 32, 3, 3, 1, 0]),
        ("relu", [True]),
        ("bn", [32]),
        ("max_pool2d", [2, 2, 0]),
        ("conv2d", [32, 32, 3, 3, 1, 0]),
        ("relu", [True]),
        ("bn", [32]),
        ("max_pool2d", [2, 1, 0]),
        ("flatten", []),
        ("linear", [meta_config["N"], 32 * 5 * 5]),
    ]

    trainer = Trainer(meta_config=meta_config, learner_config=learner_config, model_config=model_config, logger=logger)
    trainer.run_experiment()


if __name__ == "__main__":
    main()
