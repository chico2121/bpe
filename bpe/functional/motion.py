import os
import json

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

def trans_motion2d_rc(motion2d, flip, visibility=False):
    # subtract centers to local coordinates
    if visibility:
        ra_center = motion2d[2, :-1, :]
        la_center = motion2d[5, :-1, :]
        rl_center = motion2d[9, :-1, :]
        ll_center = motion2d[12, :-1, :]
        torso_center = motion2d[8, :-1, :]

        ra_motion_proj = motion2d[[2, 3, 4], :-1, :] - ra_center
        la_motion_proj = motion2d[[5, 6, 7], :-1, :] - la_center
        rl_motion_proj = motion2d[[9, 10, 11], :-1, :] - rl_center
        ll_motion_proj = motion2d[[12, 13, 14], :-1, :] - ll_center
        torso_motion_proj = motion2d[[0, 1, 2, 5, 9, 12], :-1, :] - torso_center
        
        ra_flag = motion2d[[2, 3, 4], 2, np.newaxis, :]
        la_flag = motion2d[[5, 6, 7], 2, np.newaxis, :]
        rl_flag = motion2d[[9, 10, 11], 2, np.newaxis, :]
        ll_flag = motion2d[[12, 13, 14], 2, np.newaxis, :]
        torso_flag = motion2d[[0, 1, 2, 5, 9, 12], 2, np.newaxis, :]
        
        # final flag, np.zeros((1, 1, 60)) is for velocity position: it will be detached before input the model.
        flags = np.r_[ra_flag, la_flag, rl_flag, ll_flag, torso_flag, np.zeros((1, 1, motion2d.shape[-1]))]
    
    else:
        ra_center = motion2d[2, :, :]
        la_center = motion2d[5, :, :]
        rl_center = motion2d[9, :, :]
        ll_center = motion2d[12, :, :]
        torso_center = motion2d[8, :, :]

        ra_motion_proj = motion2d[[2, 3, 4], :, :] - ra_center
        la_motion_proj = motion2d[[5, 6, 7], :, :] - la_center
        rl_motion_proj = motion2d[[9, 10, 11], :, :] - rl_center
        ll_motion_proj = motion2d[[12, 13, 14], :, :] - ll_center
        torso_motion_proj = motion2d[[0, 1, 2, 5, 9, 12], :, :] - torso_center
        
        flags = None
        
    # adding velocity
    velocity = np.c_[np.zeros((2, 1)), torso_center[:, 1:] - torso_center[:, :-1]].reshape(1, 2, -1)

    if flip:
        ra_motion_proj[:, 0, :] = -ra_motion_proj[:, 0, :]
        la_motion_proj[:, 0, :] = -la_motion_proj[:, 0, :]
        rl_motion_proj[:, 0, :] = -rl_motion_proj[:, 0, :]
        ll_motion_proj[:, 0, :] = -ll_motion_proj[:, 0, :]

        motion_proj = np.r_[la_motion_proj, ra_motion_proj, ll_motion_proj, rl_motion_proj, torso_motion_proj, velocity]
    else:
        motion_proj = np.r_[ra_motion_proj, la_motion_proj, rl_motion_proj, ll_motion_proj, torso_motion_proj, velocity]

    return motion_proj, flags # return shape: (19, 2, 64)


def trans_motion2d_rc_all_joints(motion2d, flip, visibility=False):
    # subtract centers to local coordinates
    if visibility:
        ra_center = motion2d[2, :-1, :]
        la_center = motion2d[5, :-1, :]
        rl_center = motion2d[9, :-1, :]
        ll_center = motion2d[12, :-1, :]
        torso_center = motion2d[8, :-1, :]
        velocity = np.c_[np.zeros((2, 1)), torso_center[:, 1:] - torso_center[:, :-1]].reshape(1, 2, -1)

        ra_motion_proj = motion2d[:, :-1, :] - ra_center
        la_motion_proj = motion2d[:, :-1, :] - la_center
        rl_motion_proj = motion2d[:, :-1, :] - rl_center
        ll_motion_proj = motion2d[:, :-1, :] - ll_center
        torso_motion_proj = motion2d[:, :-1, :] - torso_center
        
        motion_flag = motion2d[:, 2, np.newaxis, :]
        
        # final flag, np.zeros((1, 1, 60)) is for velocity position: it will be detached before input the model.
        flags = np.concatenate((np.concatenate([motion_flag] * 5), np.zeros((1, 1, motion2d.shape[-1]))), axis=0)
    
    else:
        ra_center = motion2d[2, :, :]
        la_center = motion2d[5, :, :]
        rl_center = motion2d[9, :, :]
        ll_center = motion2d[12, :, :]
        torso_center = motion2d[8, :, :]
        velocity = np.c_[np.zeros((2, 1)), torso_center[:, 1:] - torso_center[:, :-1]].reshape(1, 2, -1)

        ra_motion_proj = motion2d - ra_center
        la_motion_proj = motion2d - la_center
        rl_motion_proj = motion2d - rl_center
        ll_motion_proj = motion2d - ll_center
        torso_motion_proj = motion2d - torso_center
        
        flags = None
    
    if flip:
        ra_motion_proj[:, 0, :] = -ra_motion_proj[:, 0, :]
        la_motion_proj[:, 0, :] = -la_motion_proj[:, 0, :]
        rl_motion_proj[:, 0, :] = -rl_motion_proj[:, 0, :]
        ll_motion_proj[:, 0, :] = -ll_motion_proj[:, 0, :]

        motion_proj = np.r_[la_motion_proj, ra_motion_proj, ll_motion_proj, rl_motion_proj, torso_motion_proj, velocity]
    else:
        motion_proj = np.r_[ra_motion_proj, la_motion_proj, rl_motion_proj, ll_motion_proj, torso_motion_proj, velocity]

    return motion_proj, flags


def trans_motion_inv_rc(motion, sx=256, sy=256, velocity=None):
    if velocity is None:
        velocity = motion[-1].copy()
    torso = motion[-7:-1, :, :]
    ra = motion[:3, :, :] + torso[2, :, :]
    la = motion[3:6, :, :] + torso[3, :, :]
    rl = motion[6:9, :, :] + torso[4, :, :]
    ll = motion[9:12, :, :] + torso[5, :, :]
    motion_inv = np.r_[torso[:2], ra, la, np.zeros((1, 2, motion.shape[-1])), rl, ll]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += np.array([[sx], [sy]])

    return motion_inv + centers.reshape((1, 2, -1))


def normalize_motion(motion, mean_pose, std_pose):
    """
    :param motion: (J, 2, T)
    :param mean_pose: (J, 2)
    :param std_pose: (J, 2)
    :return:
    """

    return (motion - mean_pose[:, :, np.newaxis]) / std_pose[:, :, np.newaxis]


def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]


def preprocess_motion2d_rc(motion, mean_pose, std_pose, flip=False, invisibility_augmentation=False,
                           use_all_joints_on_each_bp=False):
    # TODO: Scale normalization (to fit training scales)

    if use_all_joints_on_each_bp and invisibility_augmentation:
        motion_trans, flags = trans_motion2d_rc_all_joints(motion, flip=flip, visibility=True)
        motion_trans = normalize_motion(motion_trans, mean_pose, std_pose)
        motion_trans = np.concatenate((motion_trans, flags), axis=1)
        motion_trans = motion_trans.reshape((-1, motion_trans.shape[-1]))[:-1, :]

    elif use_all_joints_on_each_bp:
        motion_trans, _ = trans_motion2d_rc_all_joints(motion, flip=flip, visibility=False)
        motion_trans = normalize_motion(motion_trans, mean_pose, std_pose)
        motion_trans = motion_trans.reshape((-1, motion_trans.shape[-1]))

    elif invisibility_augmentation:
        motion_trans, flags = trans_motion2d_rc(motion, flip=flip, visibility=True)
        motion_trans = normalize_motion(motion_trans, mean_pose, std_pose)
        motion_trans = np.concatenate((motion_trans, flags), axis=1)
        motion_trans = motion_trans.reshape((-1, motion_trans.shape[-1]))[:-1, :]

    else:
        motion_trans, _ = trans_motion2d_rc(motion, flip=flip, visibility=False)
        motion_trans = normalize_motion(motion_trans, mean_pose, std_pose)
        motion_trans = motion_trans.reshape((-1, motion_trans.shape[-1]))

    return torch.Tensor(motion_trans).unsqueeze(0)


def postprocess_motion2d_rc(motion, mean_pose, std_pose, sx=256, sy=256):
    motion = motion.detach().cpu().numpy()[0].reshape(-1, 2, motion.shape[-1])
    motion = trans_motion_inv_rc(normalize_motion_inv(motion, mean_pose, std_pose), sx, sy)
    return motion


def invisbility_aug_func(motion_proj, invisible_joints, all_joints_on_each_bp=False):
    motion_proj = np.insert(motion_proj, 2, 1, axis=1)

    max_invis_joints = np.random.randint(invisible_joints + 1)
    input_frame_length = motion_proj.shape[-1]
    K = 15  # num_of_joints
    flag = np.ones((K, 3, input_frame_length))

    target_indices = np.indices((input_frame_length, max_invis_joints))[0]
    joint_selector = np.random.randint(K, size=(input_frame_length, max_invis_joints))
    flag[joint_selector, :, target_indices] = 0

    if all_joints_on_each_bp:
        invisible_joints = np.r_[flag, flag, flag, flag, flag, np.ones((1, 3, motion_proj.shape[-1]))]
    else:
        invisible_joints = np.r_[flag[2:5], flag[5:8], flag[9:12], flag[12:15], flag[[0, 1, 2, 5, 9, 12]], np.ones(
            (1, 3, motion_proj.shape[-1]))]
    motion_proj = motion_proj * invisible_joints
    motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))[:-1, :]

    return motion_proj


def rotation_matrix_along_axis(x, angle):
    cx = np.cos(angle)
    sx = np.sin(angle)
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)
    return mat33_x


def openpose2motion(json_dir, scale=1.0, smooth=True, max_frame=None):
    json_files = sorted(os.listdir(json_dir))
    length = max_frame if max_frame is not None else len(json_files) // 8 * 8
    json_files = json_files[:length]
    json_files = [os.path.join(json_dir, x) for x in json_files]

    motion = []
    for path in json_files:
        with open(path) as f:
            jointDict = json.load(f)
            joint = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
            if len(motion) > 0:
                joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
            motion.append(joint)

    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]

    motion = np.stack(motion, axis=2)
    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion


def cocopose2motion(num_joints, json_dir, scale=1.0, smooth=True, max_frame=None, visibility=False, mean_height=None, limit=-1):
    motion = []
    with open(json_dir) as f:
        jointDict = json.load(f)
        # case for cv-api directly using
        annotations = jointDict['annotations'] if 'annotations' in jointDict else jointDict['result']['annotations']

        for annotation in annotations[:max(limit, len(annotations))]:
            if len(annotation['objects']) == 0:
                if visibility:
                    motion.append(np.zeros((num_joints, 3)))
                else:
                    motion.append(np.zeros((num_joints, 2)))
                continue

            keypoint = annotation['objects'][0]['keypoints']
            if len(keypoint) == 51:
                coco = np.array(keypoint).reshape((-1, 3))
            else:
                coco = np.array(keypoint).reshape((-1, 4))
            if visibility:
                coco = coco[:, :3]
            else:
                coco = coco[:, :2]

            nose = coco[0]
            right_shoulder = coco[6]
            right_elbow = coco[8]
            right_wrist = coco[10]
            left_shoulder = coco[5]
            left_elbow = coco[7]
            left_wrist = coco[9]
            right_hip = coco[12]
            right_knee = coco[14]
            right_ankle = coco[16]
            left_hip = coco[11]
            left_knee = coco[13]
            left_ankle = coco[15]
            neck = (right_shoulder + left_shoulder) / 2
            mid_hip = (right_hip + left_hip) / 2
            
            joint = np.array([nose,
                              neck,
                              right_shoulder,
                              right_elbow,
                              right_wrist,
                              left_shoulder,
                              left_elbow,
                              left_wrist,
                              mid_hip,
                              right_hip,
                              right_knee,
                              right_ankle,
                              left_hip,
                              left_knee,
                              left_ankle,
                              ])
            
            joint = joint.reshape((-1, 3)) if visibility else joint.reshape((-1, 2))

            if not visibility and len(motion) > 0:
                joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
            motion.append(joint)
    
    if not visibility:
        for i in range(len(motion) - 1, 0, -1):
            motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]
    motion = np.array(motion)
    motion = np.stack(motion, axis=2)
    if smooth:
        if visibility:
            smooth_motion = np.zeros_like(motion)
            for j_idx in range(len(motion)):
                joint_motion = motion[j_idx]
                joint_motion_visible = np.where(joint_motion[2] != 0)
                smooth_motion[j_idx][0, joint_motion_visible] = gaussian_filter1d(joint_motion[0, joint_motion_visible], sigma=2, axis=-1)
                smooth_motion[j_idx][1, joint_motion_visible] = gaussian_filter1d(joint_motion[1, joint_motion_visible], sigma=2, axis=-1)
                smooth_motion[j_idx][2] = joint_motion[2] 
                
            motion = smooth_motion
        else:
            motion = gaussian_filter1d(motion, sigma=2, axis=-1)
            
    if mean_height is not None:
        avg_ankle_y = (motion[11, 1, :] + motion[14, 1, :]) / 2
        nose_y = motion[0, 1, :]
        height_pixel_frame = avg_ankle_y - nose_y
        height_pixel_frame = height_pixel_frame[np.where(height_pixel_frame != 0)]
        height_pixel = np.percentile(height_pixel_frame, 90)
        motion = motion * mean_height / height_pixel
    else:  
        motion = motion * scale
    return motion


def ntupose2motion(json_dir, scale=1.0, smooth=True, max_frame=None):
    # for 171418_2 (new one)
    motion = []
    with open(json_dir) as f:
        jointDict = json.load(f)
        frames = jointDict.keys()
        length = min(max_frame, len(frames)) if max_frame is not None else len(frames)

        for i in range(length):

            frame_num = str(i + 1)
            # if '2d' not in jointDict[frame_num].keys():
            #     continue
            frame_joints = np.array(jointDict[frame_num]['2d'])

            nose = frame_joints[3]  # in NTU, not nose but head. are they different?
            right_shoulder = frame_joints[8]
            right_elbow = frame_joints[9]
            right_wrist = frame_joints[10]
            left_shoulder = frame_joints[4]
            left_elbow = frame_joints[5]
            left_wrist = frame_joints[6]
            right_hip = frame_joints[16]
            right_knee = frame_joints[17]
            right_ankle = frame_joints[18]
            left_hip = frame_joints[12]
            left_knee = frame_joints[13]
            left_ankle = frame_joints[14]
            neck = (right_shoulder + left_shoulder) / 2
            mid_hip = (right_hip + left_hip) / 2

            joint = np.array([nose,
                              neck,
                              right_shoulder,
                              right_elbow,
                              right_wrist,
                              left_shoulder,
                              left_elbow,
                              left_wrist,
                              mid_hip,
                              right_hip,
                              right_knee,
                              right_ankle,
                              left_hip,
                              left_knee,
                              left_ankle,
                              ]).reshape((-1, 2))

            if len(motion) > 0:
                joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
            motion.append(joint)

    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]
    motion = np.array(motion)
    motion = np.stack(motion, axis=2)
    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion
