"""
Ego-Exo4D feature dataset.

Expects pre-extracted per-clip features saved as .npz files under
`cfg.presave_path/<prefix>_<view>/<split>/`, one directory per view
(view in "ego", "exo1", "exo2", "exo3", "exo4"), where `<prefix>` is
the last path component of `cfg.presave_path` (e.g. "k400", "k600",
"howto100m") and all views share the same filenames.

Each .npz stores:
  a: feature tensor, b: proficiency label, c: raw sample/action name
"""
import os
import torch
from glob import glob
import numpy as np

action_list = ["Dance", "basketball", "dance", "Piano", "Violin", "soccer",
               "bouldering", "Guitar", "Cooking", "music", "cooking",
               "guitar", "piano"]
action_list_lower = {}
for i, v in enumerate(set([j.lower() for j in action_list])):
    action_list_lower[v] = i


class EgoExo4d(torch.utils.data.Dataset):
    def __init__(self, args, split="train", anno_path=""):
        self.args = args
        self.split = split
        self.presave_path = args.presave_path
        self.prefix = os.path.basename(os.path.normpath(self.presave_path))
        self.dataset = glob(os.path.join(self.presave_path, f"{self.prefix}_ego", split, "*"))
        self.dataset = sorted(self.dataset, key=lambda x: x.split("_")[-3])

    def __len__(self):
        return len(self.dataset)

    def _view_path(self, filename, view):
        return os.path.join(self.presave_path, f"{self.prefix}_{view}", self.split, filename)

    def __getitem__(self, index):
        ego_feat_path = self.dataset[index]
        filename = os.path.basename(ego_feat_path)
        exo_feat_path = self._view_path(filename, "exo1")
        exo2_feat_path = self._view_path(filename, "exo2")
        exo3_feat_path = self._view_path(filename, "exo3")
        exo4_feat_path = self._view_path(filename, "exo4")

        ego_data = np.load(ego_feat_path)
        exo_data = np.load(exo_feat_path)["a"]
        exo2_data = np.load(exo2_feat_path)["a"]
        exo3_data = np.load(exo3_feat_path)["a"]
        exo4_data = np.load(exo4_feat_path)["a"]

        name_full = str(ego_data["c"])
        label = ego_data["b"]
        name = [action for action in action_list if action in name_full][0]
        name_onehot = action_list_lower[name.lower()]
        return (ego_data["a"], exo_data, exo2_data, exo3_data, exo4_data,
                label, name, name_onehot, ego_feat_path.split("/")[-1])
