from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold

import utils


class PlayerKFold(GroupKFold):

    def __init__(self, n_splits: int = 5, random_state: int = 1):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X=None, y=None, groups=None):
        assigned_ids = []
        valid_groups = defaultdict(list)
        utils.seed_everything(self.random_state)
        for label in y.value_counts().sort_values().index.tolist():
            ids = groups[y == label].values
            ids_left = np.setdiff1d(ids, assigned_ids)
            if len(ids_left) == 0:  # There are no ids left thus gouping is ended!
                break
            else:
                np.random.shuffle(ids_left)
                for i, id_ in enumerate(ids_left):
                    valid_groups[i % self.get_n_splits()].append(id_)
                assigned_ids += ids_left.tolist()
        self.valid_groups_ = valid_groups
        all_idx = list(range(len(groups)))
        for valid_ids in valid_groups.values():
            valid_idx = [i for i, id_ in enumerate(groups.values) if id_ in valid_ids]
            train_idx = list(set(all_idx) - set(valid_idx))
            yield train_idx, valid_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def get_valid_ids(self) -> dict:
        if hasattr(self, 'valid_groups_'):
            return self.valid_groups_
        else:
            raise AttributeError('Valid fold is determined after calling `split`')
