from multiprocessing import Pool
import os
import glob
import pickle
from itertools import repeat

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from bpe import Config
from bpe.functional.motion import invisbility_aug_func, normalize_motion


def _get_variation_and_negative_names(mot, motion_names, motion_names_split):
    # TODO optimize with regex
    variation_names = []
    negative_names = []
    mot_split = mot.split("/")
    for motion_name, motion_split in zip(motion_names, motion_names_split):
        if motion_split[:2] == mot_split[:2] and motion_split[2] != mot_split[2] and \
                motion_split[4] == mot_split[4]:
            variation_names.append(motion_name)
        elif motion_split[-1] != mot_split[-1] or motion_split[0] != mot_split[0]:
            negative_names.append(motion_name)
    return variation_names, negative_names


class _UnityDatasetBase(Dataset):
    def __init__(self, phase, config):
        super(_UnityDatasetBase, self).__init__()

        assert os.path.exists(config.data_dir), f"{config.data_dir} does not exist"
        print('Loading {} dataset'.format(phase))

        assert phase in ['train', 'test']
        self.data_root = os.path.join(config.data_dir, phase)
        self.phase = phase
        self.unit = config.unit
        self.view_angles = config.view_angles

        self.body_parts = config.body_parts
        self.body_part_names = config.body_part_names
        self.num_of_joints = config.unique_nr_joints
        self.inputs_name_for_view_learning = config.quadruplet_inputs_name_for_view_learning
        self.num_of_motions = config.num_of_motions
        self.num_of_skeletons = config.num_of_skeletons
        self.num_of_views = config.num_of_views
        self.invisibility_augmentation = config.invisibility_augmentation
        self.num_of_max_invis_joints = config.num_of_max_invis_joints
        self.use_all_joints_on_each_bp = config.use_all_joints_on_each_bp
        self.action_category_balancing = config.action_category_balancing
        if self.invisibility_augmentation:
            self.body_parts_invis = config.body_parts_invis

        if phase == 'train':
            self.character_names = ['FuseFemaleA', 'FuseFemaleB', 'FuseFemaleC',
                                    'FuseFemaleD', 'FuseFemaleE', 'FuseMaleBruteA', 'FuseMaleBruteB',
                                    'FuseMaleBruteC', 'FuseMaleBruteD', 'FuseMaleBruteE', 'FuseFemaleF',
                                    'FuseMaleBruteF']
            self.aug = True
            self.joint_noise_level = config.joint_noise_level
            self.input_frame_length = config.length_of_frames_train
            self.additional_view_aug = True
        else:
            self.character_names = ['CharB', 'CharC', 'CharD', 'CharE', 'CharF']
            self.aug = False
            self.joint_noise_level = 0.0
            self.input_frame_length = config.length_of_frames_test
            self.additional_view_aug = False

        self.motion_names, self.variations_by_motion, self.variation_and_negative_names = self._build_motion_path_items()

        if self.action_category_balancing and phase == 'train':  # for motion category balance on training set
            motion_names_adv = [i for i in self.motion_names if i[:3] == 'Adv'] * 5
            motion_names_spo = [i for i in self.motion_names if i[:3] == 'Spo'] * 5
            motion_names_com = [i for i in self.motion_names if i[:3] == 'Com']
            motion_names_dan = [i for i in self.motion_names if i[:3] == 'Dan']
            self.motion_names = motion_names_com + motion_names_dan + motion_names_adv + motion_names_spo

        # self.mean_pose_bpe, self.std_pose_bpe, self.mean_pose_unified, self.std_pose_unified = self.get_meanpose(config)
        self.mean_pose_bpe, self.std_pose_bpe = self.get_meanpose(config)
        if self.use_all_joints_on_each_bp:
            self.mean_pose_all_joints_on_each_bp, self.std_pose_all_joints_on_each_bp = self.get_meanpose(config, all_joints_bp=True)
            self.body_parts_entire_body = config.body_parts_entire_body
            if self.invisibility_augmentation:
                self.body_parts_invis_entire_body = config.body_parts_invis_entire_body

        # memoization to optimize read time
        self.loaded_items = {}
        # self._preload_items()

    def _build_motion_path_items(self, use_cache=True, save_cache=True, num_workers=12):
        print('\t Building motion path items')
        glob_path = os.path.join(self.data_root, self.character_names[0], '*/*/*/motions/*.npy')

        BPE_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "training_cache")
        os.makedirs(BPE_CACHE, exist_ok=True)
        MOTION_CACHE_PATH = os.path.join(BPE_CACHE, "bpe_cache_%s" % '_'.join(glob_path.split("/")[-7:]))
        MOTION_CACHE_PATH_MOTION_NAMES = MOTION_CACHE_PATH + "_motion_names.pickle"
        MOTION_CACHE_PATH_VARIATION_NAMES = MOTION_CACHE_PATH + "_variation_names.pickle"
        MOTION_CACHE_PATH_VAR_NEG_NAMES = MOTION_CACHE_PATH + "_var_neg_names.pickle"

        if use_cache and os.path.isfile(MOTION_CACHE_PATH_MOTION_NAMES):
            print('\t [Cache Found] motion path items: ' + MOTION_CACHE_PATH)
            with open(MOTION_CACHE_PATH_MOTION_NAMES, "rb") as fr:
                motion_names = pickle.load(fr)
            with open(MOTION_CACHE_PATH_VARIATION_NAMES, "rb") as fr:
                variations_by_motion = pickle.load(fr)
            with open(MOTION_CACHE_PATH_VAR_NEG_NAMES, "rb") as fr:
                variation_and_negative_names = pickle.load(fr)
        else:
            print('\t [Cache Not Found] motion path items')
            items = glob.glob(glob_path)

            motion_names_split = [x.split('/')[-5:] for x in items]
            motion_names = ['/'.join(x) for x in motion_names_split]
            variations_by_motion = {
                motion_name: [float(var.split("_")[1]) for var in motion_name.split("/")[2].split("|")]
                for motion_name in tqdm(motion_names)
            }
            assert len(motion_names) != 0, f"Dataset seems to be incomplete or corrupted. " \
                                           f"Possible issue: https://github.com/chico2121/SARA_Dataset/issues/2"

            pool = Pool(processes=num_workers)
            variation_and_negative_names_list = pool.starmap(_get_variation_and_negative_names,
                                                             tqdm(zip(motion_names, repeat(motion_names),
                                                                      repeat(motion_names_split)),
                                                                  total=len(motion_names)))

            # disregard samples with no variations
            motion_names_filtered = []
            variations_by_motion_filtered = {}
            variation_and_negative_names = {}
            for motion_name, v_n_item in zip(motion_names, variation_and_negative_names_list):
                if len(v_n_item[0]) != 0:
                    motion_names_filtered.append(motion_name)
                    variation_and_negative_names[motion_name] = v_n_item
                    variations_by_motion_filtered[motion_name] = variations_by_motion[motion_name]
            motion_names = motion_names_filtered
            variations_by_motion = variations_by_motion_filtered

            if save_cache:
                with open(MOTION_CACHE_PATH_MOTION_NAMES, "wb") as fw:
                    pickle.dump(motion_names, fw)
                with open(MOTION_CACHE_PATH_VARIATION_NAMES, "wb") as fw:
                    pickle.dump(variations_by_motion, fw)
                with open(MOTION_CACHE_PATH_VAR_NEG_NAMES, "wb") as fw:
                    pickle.dump(variation_and_negative_names, fw)

        print('\t [Build Complete] motion path items')

        return motion_names, variations_by_motion, variation_and_negative_names

    def _preload_items(self):
        for motion in self.motion_names:
            for char in self.character_names:
                path = self.build_item(motion, char)
                if os.path.exists(path):
                    self.loaded_items[path] = np.load(path)

    def build_item(self, mot_name, char_name):
        """
        :param mot_name: animation_name/motions/xxx.npy
        :param char_name: character_name
        :return:
        """
        return os.path.join(self.data_root, char_name, mot_name)

    def gen_aug_param(self, rotate=False, invisibility=False, view_aug=False, max_noise_level=0.0, movement=False):
        if rotate:
            ret_dict = {'ratio': np.random.uniform(0.8, 1.2),
                        'roll': np.random.uniform((-np.pi / 9, -np.pi / 9, -np.pi / 6),
                                                  (np.pi / 9, np.pi / 9, np.pi / 6))}
        elif invisibility:  # maximum 5 joints are tagged 'invisible'
            ret_dict = {'ratio': np.random.uniform(0.5, 1.5),
                        'invisibility_joints': self.num_of_max_invis_joints}
        else:
            ret_dict = {'ratio': np.random.uniform(0.5, 1.5)}

        # View augmentation
        if view_aug:
            ret_dict['view_aug'] = 0.0
        else:
            ret_dict['view_aug'] = 0.0

        if max_noise_level > 0.0:
            ret_dict['noise'] = max_noise_level

        if movement:
            ret_dict['movement'] = True

        return ret_dict

    @staticmethod
    def augmentation(data, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        if param is None:
            return data, param

        # rotate
        if 'roll' in param.keys():
            cx, cy, cz = np.cos(param['roll'])
            sx, sy, sz = np.sin(param['roll'])
            mat33_x = np.array([
                [1, 0, 0],
                [0, cx, -sx],
                [0, sx, cx]
            ], dtype='float')
            mat33_y = np.array([
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy]
            ], dtype='float')
            mat33_z = np.array([
                [cz, -sz, 0],
                [sz, cz, 0],
                [0, 0, 1]
            ], dtype='float')
            data = mat33_x @ mat33_y @ mat33_z @ data

        # scale
        if 'ratio' in param.keys():
            data = data * param['ratio']

        if 'movement' in param.keys():
            aug_data = np.zeros_like(data)

            aug_data[:, :, 0] = data[:, :, 0]
            for i in range(1, data.shape[-1]):
                mix_prob = np.clip(np.abs(0.2 * np.random.randn()), 0.0, 0.6)  # [0.0, 0.6]. Max prob. @ 0.0
                aug_data[:, :, i] = data[:, :, i] * (1 - mix_prob) + data[:, :, i-1] * mix_prob

            data = aug_data

        if 'noise' in param.keys():
            max_noise_level = param['noise']
            noise_level = np.random.uniform(0, max_noise_level)
            noise_data = noise_level * np.random.randn(*data.shape)

            data += noise_data

        return data, param

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.motion_names) * len(self.character_names)

    def get_local3d(self, motion3d, angles=None):
        """
        Get the unit vectors for local rectangular coordinates for given 3D motion
        :param motion3d: numpy array. 3D motion from 3D joints positions, shape (nr_joints, 3, nr_frames).
        :param angles: tuple of length 3. Rotation angles around each axis.
        :return: numpy array. unit vectors for local rectangular coordinates's , shape (3, 3).
        """
        # 6 RightArm 5 LeftArm 12 RightUpLeg 11 LeftUpLeg
        horizontal = (motion3d[6] - motion3d[5] + motion3d[12] - motion3d[11]) / 2
        horizontal = np.mean(horizontal, axis=1)
        horizontal = horizontal / np.linalg.norm(horizontal)
        local_y = np.array([0, 1, 0])
        local_z = np.cross(local_y, horizontal)  # bugs!!!, horizontal and local_Z may not be perpendicular
        local_z = local_z / np.linalg.norm(local_z)
        local_x = np.cross(local_y, local_z)
        local = np.stack([local_x, local_y, local_z], axis=0)

        if angles is not None:
            local = self.rotate_coordinates(local, angles)

        return local

    def rotate_coordinates(self, local3d, angles):
        """
        Rotate local rectangular coordinates from given view_angles.

        :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
        :param angles: tuple of length 3. Rotation angles around each axis.
        :return:
        """
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        x = local3d[0]
        x_cpm = np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ], dtype='float')
        x = x.reshape(-1, 1)
        mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

        mat33_y = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], dtype='float')

        local3d = local3d @ mat33_x.T @ mat33_y
        return local3d

    def preprocessing_rc(self, item, view_angle=None, param=None):
        """
        :param item: filename built from self.build_tiem
        :return:
        """
        # retrieve from the table if already encountered
        if item in self.loaded_items:
            motion3d = self.loaded_items[item]
        else:
            motion3d = np.load(item)
            self.loaded_items[item] = motion3d

        if self.aug:
            motion3d, param = self.augmentation(motion3d, param)

        # convert 3d to 2d
        local3d = None
        if view_angle is not None:
            if getattr(param, 'view_aug', 0.0) > 0.0:
                # Augment view angle (pitch, yaw, roll)
                max_aug_degree = param['view_aug']
                view_angle = tuple([v + np.random.uniform(-np.pi * max_aug_degree / 180, + np.pi * max_aug_degree / 180) for v in view_angle])

            local3d = self.get_local3d(motion3d, view_angle)

        motion_proj_all_joints, motion_proj_bp = self.trans_motion3d_rc(motion3d, local3d, self.unit)

        # no aug and empty invisibility options
        if param is None:
            param = {}
            param['invisibility_joints'] = 0

        if self.use_all_joints_on_each_bp:
            motion_proj_all_joints = normalize_motion(motion_proj_all_joints, self.mean_pose_all_joints_on_each_bp,
                                                      self.std_pose_all_joints_on_each_bp)
            motion_proj_bp = normalize_motion(motion_proj_bp, self.mean_pose_bpe, self.std_pose_bpe)

            if self.invisibility_augmentation:
                invisible_joints = param['invisibility_joints']
                motion_proj_all_joints_invis = invisbility_aug_func(motion_proj_all_joints.copy(), invisible_joints,
                                                                    all_joints_on_each_bp=True)
                motion_proj_invis = invisbility_aug_func(motion_proj_bp.copy(), invisible_joints)

                motion_proj_bp = motion_proj_bp.reshape(
                    (-1, motion_proj_bp.shape[-1]))  # reshape to (joints*2, len_frames)
                return {'default': motion_proj_bp, 'view_invis_input': motion_proj_invis,
                        'invis_input': motion_proj_all_joints_invis, 'all_joints_input': None}

            else:
                motion_proj_all_joints = motion_proj_all_joints.reshape((-1, motion_proj_all_joints.shape[-1]))  # reshape to (joints*2, len_frames)
                motion_proj_bp = motion_proj_bp.reshape((-1, motion_proj_bp.shape[-1]))
                return {'default': motion_proj_bp, 'view_invis_input': None,
                        'invis_input': None, 'all_joints_input': motion_proj_all_joints}
        else:
            motion_proj_bp = normalize_motion(motion_proj_bp, self.mean_pose_bpe, self.std_pose_bpe)

            if self.invisibility_augmentation:
                motion_proj_invis = invisbility_aug_func(motion_proj_bp.copy(), param['invisibility_joints'])

                motion_proj_bp = motion_proj_bp.reshape((-1, motion_proj_bp.shape[-1]))  # reshape to (joints*2, len_frames)
                return {'default': motion_proj_bp, 'invis_input': motion_proj_invis}

            else:
                motion_proj_bp = motion_proj_bp.reshape((-1, motion_proj_bp.shape[-1]))  # reshape to (joints*2, len_frames)
                return {'default': motion_proj_bp, 'invis_input': None}

    def trans_motion3d_rc(self, motion3d, local3d=None, unit=128):

        motion3d = motion3d * unit

        # orthonormal projection
        if local3d is not None:
            motion_proj = local3d[[0, 1], :] @ motion3d  # (17, 2, 64)
        else:
            motion_proj = motion3d[:, [0, 1], :]  # (17, 2, 64)

        motion_proj[:, 1, :] = - motion_proj[:, 1, :]

        # body parts
        motion_proj_nose = motion_proj[0, :, :]
        motion_proj_neck = (motion_proj[5, :, :] + motion_proj[6, :, :]) / 2
        motion_proj_ra = motion_proj[[6, 8, 10], :, :]
        motion_proj_la = motion_proj[[5, 7, 9], :, :]
        motion_proj_hip = (motion_proj[11, :, :] + motion_proj[12, :, :]) / 2
        motion_proj_rl = motion_proj[[12, 14, 16], :, :]
        motion_proj_ll = motion_proj[[11, 13, 15], :, :]

        # delete the unused head joints and reordering along the body parts
        motion_proj = np.r_[[motion_proj_nose], [motion_proj_neck], motion_proj_ra,
                            motion_proj_la, [motion_proj_hip], motion_proj_rl, motion_proj_ll]

        motion_proj_all_joints, motion_proj_bp = self.trans_motion2d_rc(motion_proj)

        return motion_proj_all_joints, motion_proj_bp

    def trans_motion2d_rc(self, motion2d):
        # subtract centers to local coordinates
        ra_center = motion2d[2, :, :]
        la_center = motion2d[5, :, :]
        rl_center = motion2d[9, :, :]
        ll_center = motion2d[12, :, :]
        torso_center = motion2d[8, :, :]

        ra_motion_proj_bp = motion2d[[2, 3, 4], :, :] - ra_center
        la_motion_proj_bp = motion2d[[5, 6, 7], :, :] - la_center
        rl_motion_proj_bp = motion2d[[9, 10, 11], :, :] - rl_center
        ll_motion_proj_bp = motion2d[[12, 13, 14], :, :] - ll_center
        torso_motion_proj_bp = motion2d[[0, 1, 2, 5, 9, 12], :, :] - torso_center

        # adding velocity
        velocity = np.c_[np.zeros((2, 1)), torso_center[:, 1:] - torso_center[:, :-1]].reshape(1, 2, -1)

        if self.use_all_joints_on_each_bp:
            ra_motion_proj_all_joints = motion2d - ra_center
            la_motion_proj_all_joints = motion2d - la_center
            rl_motion_proj_all_joints = motion2d - rl_center
            ll_motion_proj_all_joints = motion2d - ll_center
            torso_motion_proj_all_joints = motion2d - torso_center

            motion_proj_all_joints = np.r_[
                ra_motion_proj_all_joints, la_motion_proj_all_joints, rl_motion_proj_all_joints, ll_motion_proj_all_joints,
                torso_motion_proj_all_joints, velocity
            ]  # shape : (76, 2, num_of_frames)
            motion_proj_bp = np.r_[ra_motion_proj_bp, la_motion_proj_bp, rl_motion_proj_bp, ll_motion_proj_bp, torso_motion_proj_bp, velocity]  # shape : (19, 2, num_of_frames)

        else:
            motion_proj_bp = np.r_[ra_motion_proj_bp, la_motion_proj_bp, rl_motion_proj_bp, ll_motion_proj_bp, torso_motion_proj_bp, velocity]  # shape : (19, 2, num_of_frames)
            motion_proj_all_joints, motion_proj_bp = None, motion_proj_bp

        return motion_proj_all_joints, motion_proj_bp  # return shape: (76, 2, num_of_frames) or (19, 2, num_of_frames)

    def get_meanpose(self, config, all_joints_bp=False):
        meanpose_rc_path = config.meanpose_rc_all_joints_on_each_bp_path if all_joints_bp else config.meanpose_rc_path
        stdpose_rc_path = config.stdpose_rc_all_joints_on_each_bp_path if all_joints_bp else config.stdpose_rc_path

        if os.path.exists(meanpose_rc_path) and os.path.exists(stdpose_rc_path):
            meanpose_rc = np.load(meanpose_rc_path)
            stdpose_rc = np.load(stdpose_rc_path)
        else:
            meanpose_rc, stdpose_rc = self.gen_meanpose_rc(config, all_joints_bp=all_joints_bp)
            np.save(meanpose_rc_path, meanpose_rc)
            np.save(stdpose_rc_path, stdpose_rc)
            print("meanpose_rc saved at {}".format(meanpose_rc_path))
            print("stdpose_rc saved at {}".format(stdpose_rc_path))

        return meanpose_rc, stdpose_rc

    def gen_meanpose_rc(self, config, all_joints_bp=False):
        all_paths = glob.glob(os.path.join(config.data_dir, 'train', '*/*/*/*/motion.npy'))
        all_joints = []

        for idx, path in enumerate(tqdm(all_paths, desc='gen_meanpose_rc', ncols=80)):
            var = path.split("/")[-2].split("|")
            var_parms = [float(v.split("_")[-1]) for v in var]
            if [float(0)] * len(var_parms) != var_parms:
                continue
            motion3d = np.load(path)
            local3d = None
            if config.view_angles is None:
                motion_proj = self.trans_motion3d_rc(motion3d, local3d)
                all_joints.append(motion_proj[0]) if all_joints_bp else all_joints.append(motion_proj[1])
            else:
                for angle in config.view_angles:
                    local3d = self.get_local3d(motion3d, angle)
                    motion_proj = self.trans_motion3d_rc(motion3d.copy(), local3d)
                    all_joints.append(motion_proj[0]) if all_joints_bp else all_joints.append(motion_proj[1])

        all_joints = np.concatenate(all_joints, axis=2)

        meanpose = np.mean(all_joints, axis=2)
        stdpose = np.std(all_joints, axis=2)
        stdpose[np.where(stdpose == 0)] = 1e-9
        return meanpose, stdpose

    def get_n_names_from_list(self, name_list, n=1):
        assert n != 0
        idxs = np.arange(len(name_list))
        np.random.shuffle(idxs)
        n_names = [name_list[idx] for idx in idxs[:n]]
        assert (len(n_names) == n)
        return n_names

    def preprocess_inputs_util(self, motions_list, character_names, view_names):
        # get paths to appropriate files
        paths_to_items = []
        for character in character_names:
            path_to_items = []
            for motion in motions_list:
                path = self.build_item(motion, character)
                if os.path.exists(path):
                    path_to_items.append(path)
                else:
                    print(f"{path} does not exist. Dataset seems to be incomplete or corrupted")
            paths_to_items.append(path_to_items)

        return self.preprocess_inputs_quad(paths_to_items, view_names, self.preprocessing_rc)

    def preprocess_inputs_quad(self, path_to_items, view_names, preproc_fun):
        # get augmentations
        if self.aug:
            param1 = self.gen_aug_param(rotate=False, invisibility=self.invisibility_augmentation,
                                        view_aug=self.additional_view_aug,
                                        max_noise_level=self.joint_noise_level, movement=False)
            param2 = self.gen_aug_param(rotate=False, invisibility=self.invisibility_augmentation,
                                        view_aug=self.additional_view_aug,
                                        max_noise_level=self.joint_noise_level, movement=False)
        else:
            param1 = param2 = None

        # preprocessing
        inputs = {}
        for body_part_idx, key in enumerate(self.body_part_names):
            inputs[key] = np.zeros((self.num_of_motions, self.num_of_views, self.num_of_skeletons,
                                    len(self.body_parts[body_part_idx]), self.input_frame_length))
            if self.use_all_joints_on_each_bp:
                inputs[key + '_all_joints'] = np.zeros((self.num_of_motions, self.num_of_views, self.num_of_skeletons,
                                                        len(self.body_parts_entire_body[body_part_idx]), self.input_frame_length))
                if self.invisibility_augmentation:
                    inputs[key + '_all_joints_invis'] = np.zeros((self.num_of_motions, self.num_of_views, self.num_of_skeletons,
                                                                  len(self.body_parts_invis_entire_body[body_part_idx]), self.input_frame_length))
            else:
                if self.invisibility_augmentation:
                    inputs[key + '_invis'] = np.zeros((self.num_of_motions, self.num_of_views, self.num_of_skeletons,
                                                       len(self.body_parts_invis[body_part_idx]),
                                                       self.input_frame_length))

        skel_idx = 0
        for param, paths_to_item in zip([param1, param2], path_to_items):
            for v_idx, view in enumerate(view_names):
                for mot_idx, path_to_item in enumerate(paths_to_item):
                    preprocessed_val_dict = preproc_fun(path_to_item, view, param)
                    inputs_invis_postfix = '' if not self.invisibility_augmentation else '_invis'
                    inputs_all_joints_postfix = '' if not self.use_all_joints_on_each_bp else '_all_joints'
                    dict_input_key = 'invis_input' if self.invisibility_augmentation else 'all_joints_input'
                    for body_part_idx, key in enumerate(self.body_part_names):
                        inputs[key][mot_idx][skel_idx][v_idx] = \
                            preprocessed_val_dict['default'][self.body_parts[body_part_idx], :]
                        if self.use_all_joints_on_each_bp:
                            if self.invisibility_augmentation:
                                inputs[key + inputs_all_joints_postfix + inputs_invis_postfix][mot_idx][skel_idx][v_idx] = \
                                    preprocessed_val_dict[dict_input_key][self.body_parts_invis_entire_body[body_part_idx], :]
                            else:
                                inputs[key + inputs_all_joints_postfix + inputs_invis_postfix][mot_idx][skel_idx][v_idx] = \
                                    preprocessed_val_dict[dict_input_key][self.body_parts_entire_body[body_part_idx], :]
                        else:
                            if self.invisibility_augmentation:
                                inputs[key + inputs_all_joints_postfix + inputs_invis_postfix][mot_idx][skel_idx][v_idx] = \
                                    preprocessed_val_dict[dict_input_key][self.body_parts_invis[body_part_idx], :]
                    if mot_idx == 0:
                        mot = 'p'
                    elif mot_idx == 1:
                        mot = 'sp'
                    else:
                        mot = 'n'
                    combination = f'{mot}_{skel_idx + 1}_{v_idx + 1}'
                    if combination in self.inputs_name_for_view_learning:
                        if self.invisibility_augmentation and self.use_all_joints_on_each_bp:
                            inputs[combination + inputs_invis_postfix] = preprocessed_val_dict['view_invis_input']
                        elif self.invisibility_augmentation and not self.use_all_joints_on_each_bp:
                            inputs[combination + inputs_invis_postfix] = preprocessed_val_dict['invis_input']
                        else:
                            inputs[combination] = preprocessed_val_dict['default']

            skel_idx += 1
        return inputs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
    args = parser.parse_args()

    config = Config(args)
    a = _UnityDatasetBase('train', config)
