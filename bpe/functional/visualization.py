import math
from subprocess import Popen
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm
import imageio as io


def get_video_from_sequence(seq, height, width):
    num_frames = seq.shape[-1]
    channel = 3
    color = [int(i) for i in np.random.randint(0, 255, size=channel)]
    video = np.zeros((num_frames, height, width, channel))
    for frame_idx in range(num_frames):
        cur_img = np.zeros((height, width, len(color)))
        for joint_idx in range(seq.shape[0]):
            x_coord, y_coord = seq[joint_idx, :, frame_idx]
            cv2.circle(cur_img, (int(x_coord), int(y_coord)), 1, color, 2)
        video[frame_idx] = cur_img
    return video


def get_colors_per_joint(motion_similarity_per_window, percentage_processed, thresh):
    color_per_joint = np.tile([0, 255, 0], (15, 1))

    temporal_idx = math.floor(percentage_processed * len(motion_similarity_per_window))
    if temporal_idx >= len(motion_similarity_per_window):
        return color_per_joint
    '''
    joints order : 
    [nose, neck,
    right_shoulder, right_elbow, right_wrist,
    left_shoulder, left_elbow, left_wrist,
    mid_hip,
    right_hip, right_knee, right_ankle,
    left_hip, left_knee, left_ankle]
    '''
    for bp_idx, bp in enumerate(motion_similarity_per_window[temporal_idx].keys()):
        similarity = round(motion_similarity_per_window[temporal_idx][bp], 2)
        cur_joint_color = (0, 255, 0) if similarity > thresh else (255, 0, 0)
        if bp == 'torso':
            color_per_joint[[0, 1, 8]] = cur_joint_color
        elif bp == 'ra':
            color_per_joint[[2, 3, 4]] = cur_joint_color
        elif bp == 'la':
            color_per_joint[[5, 6, 7]] = cur_joint_color
        elif bp == 'rl':
            color_per_joint[[9, 10, 11]] = cur_joint_color
        elif bp == 'll':
            color_per_joint[[12, 13, 14]] = cur_joint_color
        else:
            raise KeyError('Wrong body part key')
    return color_per_joint


def put_similarity_score_in_video(img, motion_similarity_per_window, percentage_processed, thresh):
    temporal_idx = math.floor(percentage_processed * len(motion_similarity_per_window))
    if temporal_idx >= len(motion_similarity_per_window):
        return

    similarity_per_bp = motion_similarity_per_window[temporal_idx]
    for bp_idx, bp in enumerate(similarity_per_bp.keys()):
        similarity = round(similarity_per_bp[bp], 2)
        color = (0, 255, 0) if similarity > thresh else (255, 0, 0)
        y_coord, x_coord = 50 + 50 * bp_idx, 0
        cv2.putText(img, '{}:{}'.format(bp, similarity), (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

    # setting for paper
    # for bp_idx, bp in enumerate(similarity_per_bp.keys()):
    #    similarity = round(similarity_per_bp[bp], 2)
    #    color = (0, 255, 0) if similarity > thresh else (255, 0, 0)
    #    y_coord, x_coord = 25, 2 + 70 * bp_idx
    #    cv2.putText(img, '{}:{}'.format(bp, similarity), (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
    #                color, 1)


def preprocess_sequence(seq):
    for idx, seq_item in enumerate(seq):
        if len(seq_item) == 0:
            seq[idx] = seq[idx - 1]
        if idx > 0:
            seq_item[np.where(seq_item == 0)] = seq[idx - 1][np.where(seq_item == 0)]

    return seq


def draw_seq(img, frame_seq, color_per_joint, left_padding=0, is_connected_joints=False):
    assert len(frame_seq) == len(color_per_joint)

    if is_connected_joints:
        draw_connected_joints(img, frame_seq, color_per_joint, left_padding)

    # add joints visualization
    stickwidth = 3
    # stickwidth = 1  # setting for paper
    for joint_idx, joint_xy in enumerate(frame_seq):
        x_coord, y_coord = joint_xy

        # setting for the paper
        # x_coord = x_coord // 5
        # y_coord = y_coord // 2

        color = [int(i) for i in color_per_joint[joint_idx]]
        cv2.circle(img, (left_padding + int(x_coord), int(y_coord)), stickwidth, color, 3)


def draw_connected_joints(canvas, joints, colors, left_padding):
    # connect joints with lines
    # ([nose, neck,
    # right_shoulder, right_elbow, right_wrist,
    # left_shoulder, left_elbow, left_wrist,
    # mid_hip,
    # right_hip, right_knee, right_ankle,
    # left_hip, left_knee, left_ankle,])

    limb_seq = [[0, 1], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [9, 10], [10, 11], [12, 13], [13, 14]]

    stickwidth = 2
    # stickwidth = 1 # setting for paper
    for i in range(len(limb_seq)):
        X = (int(joints[limb_seq[i][0]][0] + left_padding), int(joints[limb_seq[i][1]][0] + left_padding))
        Y = (int(joints[limb_seq[i][0]][1]), int(joints[limb_seq[i][1]][1]))

        # setting for the paper
        # X = (int(joints[limb_seq[i][0]][0] // 5 + left_padding), int(joints[limb_seq[i][1]][0] // 5 + left_padding))
        # Y = (int(joints[limb_seq[i][0]][1] // 2), int(joints[limb_seq[i][1]][1]) // 2)

        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
        polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        color = [int(i) for i in colors[limb_seq[i][0]]]
        cv2.fillConvexPoly(canvas, polygon, color)

    return canvas


def draw_frame(img, video_frame, left_padding=0, left_position=True, privacy_on=False):
    if privacy_on:
        video_frame = np.ones_like(video_frame) * 255

    left_padding, left_end = (0, left_padding) if left_position else (left_padding, img.shape[1])

    resized_video_frame = cv2.resize(video_frame, dsize=(left_end - left_padding, img.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
    img[:, left_padding:left_end, :] = resized_video_frame[:]


def video_out(output_stream: Popen, width: int, height: int,
              sequence1: np.ndarray, sequence2: np.ndarray,
              video1: Iterable[np.ndarray], video2: Iterable[np.ndarray], left_padding: int,
              motion_similarity_per_window: list,
              is_debug: bool, thresh: float, privacy_on=False, is_connected_joints=False):
    """
    visualization of the input sequences
    :param output_stream: ffmpeg stream. rawimage stream needed
    :param width:
    :param height:
    :param sequence1: shape (#frames, #joints, #coords(e.g. x,y)
    :param sequence2: shape (N, 15, 2)
    :param video1:
    :param video2:
    :param motion_similarity_per_window:
    :return:
    """
    assert sequence1.shape[1:] == (15, 2)
    assert sequence2.shape[1:] == (15, 2)

    total_vid_length = min(len(sequence1), len(sequence2))

    for frame_idx, frame_seq1, frame_seq2 in tqdm(zip(range(total_vid_length), sequence1, sequence2),
                                                  total=total_vid_length, desc='Output video saving progress'):

        canvas = np.ones((height, width, 3), np.uint8) * 255

        if video1 is not None:
            draw_frame(canvas, next(video1), left_padding=left_padding, left_position=True, privacy_on=privacy_on)
        if video2 is not None:
            draw_frame(canvas, next(video2), left_padding=left_padding, left_position=False, privacy_on=privacy_on)

        percentage_processed = float(frame_idx) / total_vid_length
        # get colors for each joint to visialize which body parts disagree
        color_per_joint = get_colors_per_joint(motion_similarity_per_window, percentage_processed, thresh)
        if frame_seq1 is not None:
            draw_seq(canvas, frame_seq1, color_per_joint, left_padding=0, is_connected_joints=is_connected_joints)
        if frame_seq2 is not None:
            draw_seq(canvas, frame_seq2, color_per_joint, left_padding=left_padding, is_connected_joints=is_connected_joints)

        put_similarity_score_in_video(canvas, motion_similarity_per_window, percentage_processed, thresh)

        output_stream.stdin.write(canvas.tostring())
        if is_debug and frame_idx == 1000:
            break
    output_stream.stdin.close()
    output_stream.wait()


def video_out_with_imageio(output_path: str, width: int, height: int,
                           sequence1: np.ndarray, sequence2: np.ndarray,
                           video1_path: str, video2_path: str, left_padding: int, pad2: int,
                           motion_similarity_per_window: list,
                           is_debug: bool, thresh: float, privacy_on=False, is_connected_joints=False):
    """
    visualization of the input sequences
    :param output_path: output path to save the file
    :param width:
    :param height:
    :param sequence1: shape (#frames, #joints, #coords(e.g. x,y)
    :param sequence2: shape (N, 15, 2)
    :param video1:
    :param video2:
    :param motion_similarity_per_window:
    :return:
    """
    assert sequence1.shape[1:] == (15, 2)
    assert sequence2.shape[1:] == (15, 2)

    # get video readers
    video1 = io.get_reader(video1_path)
    video2 = io.get_reader(video2_path)

    # get output video writer
    out_video_writer = io.get_writer(output_path, mode='I', fps=video1.get_meta_data()['fps'])

    # align videos
    video1 = iter(video1)
    video2 = iter(video2)
    for i in range(pad2):
        next(video2)

    total_vid_length = min(len(sequence1), len(sequence2))

    for frame_idx, frame_seq1, frame_seq2 in tqdm(zip(range(total_vid_length), sequence1, sequence2),
                                                  total=total_vid_length, desc='Output video saving progress'):

        canvas = np.ones((height, width, 3), np.uint8) * 255

        if video1 is not None:
            draw_frame(canvas, next(video1), left_padding=left_padding, left_position=True, privacy_on=privacy_on)
        if video2 is not None:
            draw_frame(canvas, next(video2), left_padding=left_padding, left_position=False, privacy_on=privacy_on)

        percentage_processed = float(frame_idx) / total_vid_length
        # get colors for each joint to visialize which body parts disagree
        color_per_joint = get_colors_per_joint(motion_similarity_per_window, percentage_processed, thresh)
        if frame_seq1 is not None:
            draw_seq(canvas, frame_seq1, color_per_joint, left_padding=left_padding, left_position=True,
                     is_connected_joints=is_connected_joints)
        if frame_seq2 is not None:
            draw_seq(canvas, frame_seq2, color_per_joint, left_padding=left_padding, left_position=False,
                     is_connected_joints=is_connected_joints)

        put_similarity_score_in_video(canvas, motion_similarity_per_window, percentage_processed, thresh)

        if is_debug and frame_idx == 500:
            break

        out_video_writer.append_data(canvas.to_string())

    out_video_writer.close()
