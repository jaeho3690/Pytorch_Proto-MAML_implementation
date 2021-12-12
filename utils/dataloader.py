from torch.utils.data import DataLoader

from utils.dataset import MiniImagenetDataset


def get_dataloader(meta_config, model_config):
    if meta_config["dataset"] == "miniimagenet":
        train_loader = DataLoader(
            MiniImagenetDataset(mode="train", meta_config=meta_config),
            batch_size=model_config["inner_loop"],
            shuffle=False,
            num_workers=meta_config["num_workers"],
        )
        val_loader = DataLoader(
            MiniImagenetDataset(mode="val", meta_config=meta_config),
            batch_size=model_config["inner_loop"],
            shuffle=False,
            num_workers=meta_config["num_workers"],
        )
        test_loader = DataLoader(
            MiniImagenetDataset(mode="test", meta_config=meta_config),
            batch_size=model_config["inner_loop"],
            shuffle=False,
            num_workers=meta_config["num_workers"],
        )

    else:
        raise ValueError(f"No dataset called {meta_config['dataset']}. Check name")

    return train_loader, val_loader, test_loader
