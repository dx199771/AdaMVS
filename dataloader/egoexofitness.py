"""
EgoExoFitness feature dataset.

Expects pre-extracted per-clip features saved as .npz files under
`cfg.presave_path/<prefix>_<view>/<split>/`, one directory per view
(view in "ego", "exol", "exom", "exor" -- left/middle/right camera),
where `<prefix>` is the last path component of `cfg.presave_path`
(e.g. "k400", "k600", "howto100m") and all views share the same filenames.

Each .npz stores:
  a: feature tensor, b: proficiency label (1-indexed), c: raw sample name
"""
import os
import torch
from glob import glob
import numpy as np


class EgoExoFitness(torch.utils.data.Dataset):
    def __init__(self, args, split="train", anno_path=""):
        self.args = args
        self.split = split
        self.presave_path = args.presave_path
        self.prefix = os.path.basename(os.path.normpath(self.presave_path))
        self.dataset = glob(os.path.join(self.presave_path, f"{self.prefix}_ego", split, "*"))

    def __len__(self):
        return len(self.dataset)

    def _view_path(self, filename, view):
        return os.path.join(self.presave_path, f"{self.prefix}_{view}", self.split, filename)

    def __getitem__(self, index):
        ego_feat_path = self.dataset[index]
        filename = os.path.basename(ego_feat_path)
        exol_feat_path = self._view_path(filename, "exol")
        exom_feat_path = self._view_path(filename, "exom")
        exor_feat_path = self._view_path(filename, "exor")

        ego_data = np.load(ego_feat_path)
        exo_data = np.load(exol_feat_path)
        exo2_data = np.load(exom_feat_path)
        exo3_data = np.load(exor_feat_path)

        assert ego_data["b"] == exo_data["b"]
        label = ego_data["b"] - 1  # labels are 1-indexed in the raw annotation
        name_full = str(ego_data["c"])

        name = ""
        name_onehot = 1
        return (ego_data["a"], exo_data["a"], exo2_data["a"], exo3_data["a"],
                label, name_full, name_onehot)
