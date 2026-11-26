"""Dataset registry + DataLoader factory."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .egoexo4d import EgoExo4d
from .egoexofitness import EgoExoFitness


def worker_init_fn(worker_id):
    """Give each DataLoader worker a distinct numpy seed so augmentation /
    sampling randomness doesn't repeat across workers."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataloader(args):
    """Build the train/test DataLoaders for `args.dataset`.

    Returns a dict: {"train": DataLoader, "test": DataLoader}.
    """
    data_loaders = {}

    dataset_map = {
        "egoexo4d": EgoExo4d,
        "egoexofitness": EgoExoFitness,
    }

    if args.dataset.lower() not in dataset_map:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    DatasetClass = dataset_map[args.dataset.lower()]

    data_loader_train = DatasetClass(args, "train", anno_path=args.train_anno_path)
    data_loader_test = DatasetClass(args, "test", anno_path=args.test_anno_path)

    if args.get("gb_val_source", "test") == "holdout":
        # Hold out part of train as a validation set, stratified by label.
        # The split uses a fixed seed so every run sees the same split.
        labels = np.array([int(np.load(p)["b"]) for p in data_loader_train.dataset])
        rng = np.random.RandomState(0)
        val_idx = []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            val_idx += idx[:int(round(len(idx) * args.get("val_ratio", 0.15)))].tolist()
        val_set = set(val_idx)
        train_idx = [i for i in range(len(labels)) if i not in val_set]
        full_train = data_loader_train
        data_loader_train = torch.utils.data.Subset(full_train, train_idx)
        data_loaders["val"] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full_train, sorted(val_idx)),
            batch_size=args.bs_test, shuffle=False,
            num_workers=int(args.num_workers), pin_memory=True, drop_last=False)

    train_dataloader = torch.utils.data.DataLoader(
        data_loader_train,
        batch_size=args.bs_train,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        data_loader_test,
        batch_size=args.bs_test,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    data_loaders["train"] = train_dataloader
    data_loaders["test"] = test_dataloader

    return data_loaders
