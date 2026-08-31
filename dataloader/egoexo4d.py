"""
Ego-Exo4D feature dataset.

Expects pre-extracted per-clip features saved as .npz files under
`cfg.presave_path/<split>/`, one file per ego clip, named like
`..._ego_<id>.npz`. The matching exo-view files are found by string
substitution (`ego_` -> `exo1_` / `exo2_` / `exo3_` / `exo4_`).

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
        self.dataset = glob(os.path.join(self.args.presave_path, split, "*"))
        self.dataset = sorted(self.dataset, key=lambda x: x.split("_")[-3])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ego_feat_path = self.dataset[index]
        exo_feat_path = ego_feat_path.replace("ego_", "exo1_")
        exo2_feat_path = ego_feat_path.replace("ego_", "exo2_")
        exo3_feat_path = ego_feat_path.replace("ego_", "exo3_")
        exo4_feat_path = ego_feat_path.replace("ego_", "exo4_")

        ego_data = np.load(ego_feat_path)
        exo_data = np.load(exo_feat_path)["a"]
        exo2_data = np.load(exo2_feat_path)["a"]
        exo3_data = np.load(exo3_feat_path)["a"]

        # The 4th exo view was pre-extracted with a different backbone/root
        # per pretrain source, so its path lives under its own presave dir
        # instead of a simple "ego_"->"exo4_" swap. Point these at your own
        # feature roots.
        if self.args.pretrain == "k600":
            exo4_root = "/path/to/presave/384_exo4_rgb_10sample_swin_base_patch244_window877_kinetics600_22k_name_8clip"
            exo4_feat_path = glob(f'{exo4_root}/{self.split}/*_{"_".join(exo4_feat_path.split("/")[-1].split("_")[1:])}')[0]
        elif self.args.pretrain == "howto100":
            exo4_root = "/path/to/presave/384_ego_rgb_10sample_TimeSformer_divST_8x32_224_HowTo100M_name_8clip"
            exo4_feat_path = glob(f'{exo4_root}/{self.split}/*_{"_".join(exo4_feat_path.split("/")[-1].split("_")[1:])}')[0]
        # else (k400): exo4_feat_path already matches via the "ego_"->"exo4_" swap above.
        exo4_data = np.load(exo4_feat_path)["a"]

        name_full = str(ego_data["c"])
        label = ego_data["b"]
        name = [action for action in action_list if action in name_full][0]
        name_onehot = action_list_lower[name.lower()]
        return (ego_data["a"], exo_data, exo2_data, exo3_data, exo4_data,
                label, name, name_onehot, ego_feat_path.split("/")[-1])
