"""
EgoExoFitness feature dataset.

Expects pre-extracted per-clip features saved as .npz files under
`cfg.presave_path/<split>/`, named like `..._ego_m_<id>.npz`; the 3 exo
views ("l"eft/"m"iddle/"r"ight camera) are found by string substitution.

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
        self.dataset = glob(os.path.join(self.args.presave_path, split, "*"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ego_feat_path = self.dataset[index]
        exom_feat_path = ego_feat_path.replace("ego_m_", "exo_l_")
        exol_feat_path = ego_feat_path.replace("ego_m_", "exo_r_")
        exor_feat_path = ego_feat_path.replace("ego_m_", "exo_m_")

        ego_data = np.load(ego_feat_path)
        exo_data = np.load(exom_feat_path)
        exo2_data = np.load(exol_feat_path)
        exo3_data = np.load(exor_feat_path)

        assert ego_data["b"] == exo_data["b"]
        label = ego_data["b"] - 1  # labels are 1-indexed in the raw annotation
        name_full = str(ego_data["c"])

        name = ""
        name_onehot = 1
        return (ego_data["a"], exo_data["a"], exo2_data["a"], exo3_data["a"],
                label, name_full, name_onehot)
