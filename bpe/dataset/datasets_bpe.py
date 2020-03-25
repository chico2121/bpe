import numpy as np

from bpe.dataset.unity_base_dataset import _UnityDatasetBase


class SARADataset(_UnityDatasetBase):
    def __init__(self, phase, config):
        super(SARADataset, self).__init__(phase, config)

    def __getitem__(self, index):
        # select three motions
        idx_p = int(index / len(self.character_names))
        mot_p = self.motion_names[idx_p]
        p_variations = self.variations_by_motion[mot_p]

        variation_names, negative_names = self.variation_and_negative_names[mot_p]

        idx_sp = np.random.randint(len(variation_names))
        mot_sp = variation_names[int(idx_sp)]
        sp_variations = self.variations_by_motion[mot_sp]

        # get positive and semi-positive variation score
        variation_p_sp = self.variation_score(p_variations, sp_variations)

        # negative motion
        idx_n = np.random.randint(len(negative_names))
        mot_n = negative_names[int(idx_n)]

        # select two characters
        character_names = self.get_n_names_from_list(self.character_names, n=2)

        # select two views
        view_names = self.get_n_names_from_list(self.view_angles, n=2)

        # list of positive, semi-positive, and negative motions
        motion_names = [mot_p, mot_sp, mot_n]

        # get preprocessed inputs
        inputs_bp = self.preprocess_inputs_util(motion_names, character_names, view_names)

        # merge input and outputs
        data = {'variation_p_sp': variation_p_sp, **inputs_bp}

        return data

    @staticmethod
    def variation_score(variation_param1, variation_param2):
        MAX_VARIATION = 3
        dist_sum = np.abs(np.array(variation_param1) - np.array(variation_param2)).sum()

        return dist_sum / (2 * MAX_VARIATION)  # (2 * len(variation_param1))
