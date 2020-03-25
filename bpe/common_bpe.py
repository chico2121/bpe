import itertools
import os

import more_itertools
import torch
import numpy as np

from bpe.functional import utils
from collections import namedtuple
import multiprocessing


BodyPart = namedtuple('BodyPart', [
    'right_arm',
    'left_arm',
    'right_leg',
    'left_leg',
    'torso',  # 6 joints + velocity
])


class Config:
    name = None
    device = None

    # data paths
    data_dir = None
    meanpose_path = None
    stdpose_path = None
    meanpose_rc_path = None
    stdpose_rc_path = None

    # training paths
    save_dir = './train_log'
    exp_dir = None
    log_dir = None
    model_dir = None

    # data info
    img_size = (512, 512)
    unit = 128  # TODO: more descriptive variable name
    unique_nr_joints = 15
    view_angles = [(np.pi * pitch_ang / 8.0, np.pi * yaw_ang / 2.0, 0)
                   for pitch_ang in np.arange(-0.5, 0.5001, 0.5) for yaw_ang in np.arange(-1.0, 1.001, 0.25)]

    # order of names is important, modify at your own risk2
    num_of_motions = 3  # positive, semi_positive, negative
    num_of_skeletons = 2
    num_of_views = 2
    length_of_frames_train = 32
    length_of_frames_test = 32

    # inputs idx for view embedding learning : e.g. combination of positive, first skeleton idx, first view idx
    quadruplet_inputs_name_for_view_learning = ["p_1_1", "p_1_2", "n_2_1", "n_2_2"]

    nr_body_parts = len(BodyPart._fields)
    _nr_joints = BodyPart(3, 3, 3, 3, 7)  # BodyPartWithVelocity(3, 3, 3, 3, 6, 1)
    velocity_xy = 2

    # training settings
    L2regular = True
    Batchnorm = True

    invisibility_augmentation = False
    num_of_max_invis_joints = 3
    triplet_distance = 'cosine'
    similarity_distance_metric = 'cosine'
    use_all_joints_on_each_bp = False
    action_category_balancing = True

    recon_weight = 1.0
    triplet_margin = 0.3  # TODO: Increase (up to 1.0)
    triplet_weight = 0.7
    quadruplet_margin = 0.5  # TODO: Increase (up to 1.0)
    quadruplet_weight = 1.0
    quadruplet_sim_weight = 1.0
    variation_control_param = 0.2
    use_footvel_loss = False
    foot_idx = None  # idx of foot in right_leg of left_leg
    footvel_loss_weight = 0.0
    motion_embedding_l2reg = True

    joint_noise_level = 0.05  # 0 -> disabled

    nr_epochs = 70
    batch_size = 2048

    num_workers = min(multiprocessing.cpu_count() - 1, 20)
    lr = 1e-3
    lr_decay_rate = 0.98
    weight_decay = 1e-2

    save_frequency = 1
    val_frequency = 8
    lr_update_frequency_per_epoch = 3

    def generate_joints_parts_idxs(self, num_channels, invis_aug=False, entire_body=False):

        len_joints = BodyPart(*(np.asarray(self._nr_joints_entire_body) * num_channels)) if entire_body \
            else BodyPart(*(np.asarray(self._nr_joints) * num_channels))
        if invis_aug:
            len_joints = BodyPart(*(list(len_joints[:-1]) + [len_joints[-1] - 1]))  # remove visibility on velocity

        # BodyPartWithVelocity idxs for coordinates + (opt. visibility)
        body_parts = BodyPart(
            *more_itertools.split_before(range(sum(len_joints)), lambda i: i in list(itertools.accumulate(len_joints)))
        )

        return len_joints, body_parts

    def __init__(self, args):
        self.name = args.name
        self.data_dir = args.data_dir

        self.use_footvel_loss = args.use_footvel_loss if hasattr(args, 'use_footvel_loss') else False
        self.invisibility_augmentation = args.use_invisibility_aug if hasattr(args, 'use_invisibility_aug') else False

        if hasattr(args, "triplet_distance"):
            self.triplet_distance = args.triplet_distance
            self.similarity_distance_metric = args.similarity_distance_metric

        if hasattr(args, "sim_loss_weight") and args.sim_loss_weight is not None:
            self.quadruplet_sim_weight = args.sim_loss_weight

        if hasattr(args, 'norecon') and args.norecon:
            self.recon_weight = 0.0

        self.foot_idx = [4, 5]
        self.unit = 64

        len_joints, self.body_parts = self.generate_joints_parts_idxs(2)
        len_joints_decoder = len_joints  # decoder should output same #channels as without visibility aug
        self.default_body_parts = self.body_parts

        # x, y, (visibility)
        if self.invisibility_augmentation:
            len_joints, self.body_parts_invis = self.generate_joints_parts_idxs(3, invis_aug=True)
            self.default_body_parts = self.body_parts_invis

        self.use_all_joints_on_each_bp = \
            args.use_all_joints_on_each_bp if hasattr(args, 'use_all_joints_on_each_bp') else False

        if self.name == 'sim_test' and args.use_all_joints_on_each_bp:
            self.meanpose_rc_path = os.path.join(self.data_dir, "meanpose_rc_all_joints_on_each_bp_unit128.npy")
            self.stdpose_rc_path = os.path.join(self.data_dir, "stdpose_rc_all_joints_on_each_bp_unit128.npy")
        else:
            self.meanpose_rc_path = os.path.join(self.data_dir, "meanpose_rc_with_view_unit64.npy")
            self.stdpose_rc_path = os.path.join(self.data_dir, "stdpose_rc_with_view_unit64.npy")

        if self.use_all_joints_on_each_bp:
            if not self.name == 'sim_test':
                self.meanpose_rc_all_joints_on_each_bp_path = \
                    os.path.join(args.data_dir, 'meanpose_rc_all_joints_on_each_bp_unit64.npy')
                self.stdpose_rc_all_joints_on_each_bp_path = \
                    os.path.join(args.data_dir, 'stdpose_rc_all_joints_on_each_bp_unit64.npy')
            self._nr_joints_entire_body = BodyPart(self.unique_nr_joints, self.unique_nr_joints, self.unique_nr_joints,
                                                   self.unique_nr_joints, self.unique_nr_joints + 1)
            len_joints_entire_body, self.body_parts_entire_body = self.generate_joints_parts_idxs(2, entire_body=True)
            self.default_body_parts = self.body_parts_entire_body

            if self.invisibility_augmentation:
                len_joints_entire_body, self.body_parts_invis_entire_body = \
                    self.generate_joints_parts_idxs(3, invis_aug=True, entire_body=True)
                self.default_body_parts = self.body_parts_invis_entire_body

        velocity_xy = 2

        self.body_part_names = ['ra', 'la', 'rl', 'll', 'torso']

        base_channels = 16
        mot_en_arm_leg_layer2_ch = 1 * base_channels
        mot_en_arm_leg_layer3_ch = 2 * base_channels
        mot_en_arm_leg_layer4_ch = 4 * base_channels
        mot_en_torso_layer2_ch = 2 * base_channels
        mot_en_torso_layer3_ch = 4 * base_channels
        mot_en_torso_layer4_ch = 8 * base_channels

        body_en_arm_leg_layer2_ch = base_channels
        body_en_arm_leg_layer3_ch = 2 * base_channels
        body_en_arm_leg_layer4_ch = 4 * base_channels
        body_en_arm_leg_layer5_ch = base_channels
        body_en_torso_layer2_ch = base_channels
        body_en_torso_layer3_ch = 2 * base_channels
        body_en_torso_layer4_ch = 4 * base_channels
        body_en_torso_layer5_ch = 2 * base_channels

        view_en_layer2_ch = 2 * base_channels
        view_en_layer3_ch = 3 * base_channels
        view_en_layer4_ch = 4 * base_channels

        de_layer2_ch = 4 * base_channels
        de_layer3_ch = 2 * base_channels

        self.view_en_channels = [sum(len_joints) - velocity_xy, view_en_layer2_ch, view_en_layer3_ch, view_en_layer4_ch]

        if self.use_all_joints_on_each_bp:

            body_en_layer2_ch = 4 * base_channels
            body_en_layer3_ch = 6 * base_channels
            body_en_layer4_ch = 8 * base_channels

            self.mot_en_channels = BodyPart(
                [len_joints_entire_body.right_arm, mot_en_arm_leg_layer2_ch, mot_en_arm_leg_layer3_ch,
                 mot_en_arm_leg_layer4_ch],
                [len_joints_entire_body.left_arm, mot_en_arm_leg_layer2_ch, mot_en_arm_leg_layer3_ch,
                 mot_en_arm_leg_layer4_ch],
                [len_joints_entire_body.right_leg, mot_en_arm_leg_layer2_ch, mot_en_arm_leg_layer3_ch,
                 mot_en_arm_leg_layer4_ch],
                [len_joints_entire_body.left_leg, mot_en_arm_leg_layer2_ch, mot_en_arm_leg_layer3_ch,
                 mot_en_arm_leg_layer4_ch],
                [len_joints_entire_body.torso, mot_en_torso_layer2_ch, mot_en_torso_layer3_ch, mot_en_torso_layer4_ch])
            self.body_en_channels = [sum(len_joints) - velocity_xy, body_en_layer2_ch, body_en_layer3_ch,
                                     body_en_layer4_ch]
            self.de_channels = BodyPart(
                *[(mot_en_item[-1] + self.body_en_channels[-1] + self.view_en_channels[-1], de_layer2_ch, de_layer3_ch,
                   x_len_joints)
                  for mot_en_item, x_len_joints in
                  zip(self.mot_en_channels, len_joints_decoder)])

        else:
            self.mot_en_channels = BodyPart(
                [len_joints.right_arm, mot_en_arm_leg_layer2_ch, mot_en_arm_leg_layer3_ch, mot_en_arm_leg_layer4_ch],
                [len_joints.left_arm, mot_en_arm_leg_layer2_ch, mot_en_arm_leg_layer3_ch, mot_en_arm_leg_layer4_ch],
                [len_joints.right_leg, mot_en_arm_leg_layer2_ch, mot_en_arm_leg_layer3_ch, mot_en_arm_leg_layer4_ch],
                [len_joints.left_leg, mot_en_arm_leg_layer2_ch, mot_en_arm_leg_layer3_ch, mot_en_arm_leg_layer4_ch],
                [len_joints.torso, mot_en_torso_layer2_ch, mot_en_torso_layer3_ch, mot_en_torso_layer4_ch])
            self.body_en_channels = BodyPart(
                [len_joints.right_arm, body_en_arm_leg_layer2_ch, body_en_arm_leg_layer3_ch, body_en_arm_leg_layer4_ch,
                 body_en_arm_leg_layer5_ch],
                [len_joints.left_arm, body_en_arm_leg_layer2_ch, body_en_arm_leg_layer3_ch, body_en_arm_leg_layer4_ch,
                 body_en_arm_leg_layer5_ch],
                [len_joints.right_leg, body_en_arm_leg_layer2_ch, body_en_arm_leg_layer3_ch, body_en_arm_leg_layer4_ch,
                 body_en_arm_leg_layer5_ch],
                [len_joints.left_leg, body_en_arm_leg_layer2_ch, body_en_arm_leg_layer3_ch, body_en_arm_leg_layer4_ch,
                 body_en_arm_leg_layer5_ch],
                [len_joints.torso - velocity_xy, body_en_torso_layer2_ch, body_en_torso_layer3_ch,
                 body_en_torso_layer4_ch, body_en_torso_layer5_ch])
            self.de_channels = BodyPart(
                *[(mot_en_item[-1] + body_en_item[-1] + self.view_en_channels[-1], de_layer2_ch, de_layer3_ch,
                   x_len_joints)
                  for mot_en_item, body_en_item, x_len_joints in
                  zip(self.mot_en_channels, self.body_en_channels, len_joints_decoder)])

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "logdir" in args and args.logdir:
            self.save_dir = args.logdir
        self.exp_dir = os.path.join(self.save_dir, 'exp_' + self.name)
        self.log_dir = os.path.join(self.exp_dir, 'log/')
        self.model_dir = os.path.join(self.exp_dir, 'model/')
        utils.ensure_dirs([self.log_dir, self.model_dir])
