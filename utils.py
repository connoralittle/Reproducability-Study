from dataclasses import dataclass
import numpy as np

#Spatio Temporal Graph Dataset
@dataclass()
class STG_Dataset:
    adjs: np.ndarray
    adjs_timestep: np.ndarray
    feats: np.ndarray
    feats_timestep: np.ndarray
    labels: np.ndarray
    labels_timestep: np.ndarray

    n_nodes: int
    n_timestamps: int
    n_class: int
    n_feat: int

    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray