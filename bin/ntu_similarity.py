import multiprocessing as mp
import time

import math
import os
from collections import OrderedDict

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tslearn.metrics import dtw_path

from bpe import Config
from bpe.functional.motion import preprocess_motion2d_rc, cocopose2motion
from bpe.functional.utils import pad_to_height
from bpe.model import networks_bpe


def preprocess_motion2tensor(config, motions, meanpose, stdpose,
                             flip=False, visibility=False, use_all_joints_on_each_bp=False):
    if flip:
        _input = preprocess_motion2d_rc(motions, meanpose, stdpose, invisibility_augmentation=visibility,
                                        use_all_joints_on_each_bp=use_all_joints_on_each_bp)
        flipped_input = preprocess_motion2d_rc(motions, meanpose, stdpose, flip=flip,
                                               invisibility_augmentation=visibility,
                                               use_all_joints_on_each_bp=use_all_joints_on_each_bp)
        inputs = _input.to(config.device)
        flipped_inputs = flipped_input.to(config.device)
        return inputs, flipped_inputs

    else:
        _input = preprocess_motion2d_rc(motions, meanpose, stdpose, invisibility_augmentation=visibility,
                                        use_all_joints_on_each_bp=use_all_joints_on_each_bp)
        inputs = _input.to(config.device)
        return inputs, None


def find_init_point_pix(query, th_value=0.3):
    for idx in range(query.shape[-1] - 1):
        temp = []
        for idxx in range(int(query.shape[-2] / 2)):
            temp.append(
                np.linalg.norm(query[:, [idxx * 2, idxx * 2 + 1], idx] - query[:, [idxx * 2, idxx * 2 + 1], idx + 1]))
        if sum(temp) > th_value:
            return idx
    return 0


def ntu_similarity_prepart(net, x, f, y, body_parts, bps, dist, slide, window_size):
    input1 = []
    length1 = x.shape[-1]
    input2 = []
    length2 = f.shape[-1]
    input3 = []
    length3 = y.shape[-1]

    # x1: entire input
    for i in range(math.ceil(((length1 - window_size + 1) / slide))):
        input1.append(x[:, :, i * slide: i * slide + window_size])
    for j in range(math.ceil(((length2 - window_size + 1) / slide))):
        input2.append(f[:, :, j * slide: j * slide + window_size])
    for k in range(math.ceil(((length3 - window_size + 1) / slide))):
        input3.append(y[:, :, k * slide: k * slide + window_size])

    mot_encoder = net.mot_encoders
    input1 = torch.cat(input1, dim=0)
    input2 = torch.cat(input2, dim=0)
    input3 = torch.cat(input3, dim=0)

    query_o_framesPerBP = []
    query_f_framesPerBP = []
    cand_framesPerBP = []

    for bp_idx, bp in enumerate(bps):
        _x_input1 = input1[:, getattr(body_parts, bp), :]
        _x_input2 = input2[:, getattr(body_parts, bp), :]
        _x_input3 = input3[:, getattr(body_parts, bp), :]
        _x_input = torch.cat([_x_input1, _x_input2, _x_input3])
        mot_ebd = mot_encoder[bp_idx](_x_input)
        mot_ebd = mot_ebd.reshape((mot_ebd.shape[0], -1)).cpu().data.numpy()

        query_o_framesPerBP.append(mot_ebd[:_x_input1.shape[0], :])
        query_f_framesPerBP.append(mot_ebd[_x_input1.shape[0]: _x_input1.shape[0] + _x_input2.shape[0], :])
        cand_framesPerBP.append(mot_ebd[_x_input1.shape[0] + _x_input2.shape[0]:, :])

    query_arm_flipped = [query_f_framesPerBP[0], query_f_framesPerBP[1], query_o_framesPerBP[2],
                         query_o_framesPerBP[3], query_o_framesPerBP[4]]
    query_leg_flipped = [query_o_framesPerBP[0], query_o_framesPerBP[1], query_f_framesPerBP[2],
                         query_f_framesPerBP[3], query_o_framesPerBP[4]]
    querys = [query_o_framesPerBP, query_f_framesPerBP, query_arm_flipped, query_leg_flipped]

    metric = torch.nn.CosineSimilarity(dim=0, eps=1e-50) if dist == 'cosine' else torch.nn.PairwiseDistance()

    return cand_framesPerBP, metric, querys


def ntu_similarity(net, config, x, f, y, window_size=16, slide=2, visibility=False, dist='cosine',
                   use_all_joints_on_each_bp=False):
    if use_all_joints_on_each_bp:
        body_parts = config.body_parts_entire_body if not visibility else config.body_parts_invis_entire_body
    else:
        body_parts = config.body_parts if not visibility else config.body_parts_invis

    bps = list(body_parts._fields)

    cand_framesPerBP, metric, querys = ntu_similarity_prepart(net, x, f, y, body_parts, bps, dist, slide, window_size)

    finals = [[] for _ in range(4)]
    for query_idx, query in enumerate(querys):
        for bp in range(len(bps)):
            path, _ = dtw_path(query[bp], cand_framesPerBP[bp])

            similarities = []
            for i in range(len(path)):
                if dist == "cosine":
                    metric_sim = metric(torch.Tensor(query[bp][path[i][0]]),
                                        torch.Tensor(cand_framesPerBP[bp][path[i][1]])).numpy()
                else:
                    metric_sim = metric(torch.Tensor(query[bp][path[i][0]]).unsqueeze(0),
                                        torch.Tensor(cand_framesPerBP[bp][path[i][1]]).unsqueeze(0)).numpy()[0]
                similarities.append(metric_sim)

            final_sim = np.mean(similarities)
            finals[query_idx].append(final_sim)

    # origin, flipped, arm_flipped, leg_flipped
    final_sim_calculated = [np.mean(final_v) for final_v in finals]  # different paths don't `np.mean(_, axis=1)`
    if dist == 'cosine':
        return tuple(final_sim_calculated)
    else:
        return tuple([1 / item for item in final_sim_calculated])


def ntu_similarity_global_dtw(net, config, x, f, y, min_num=2, window_size=16, slide=2, visibility=False, dist='cosine',
                              use_all_joints_on_each_bp=False):
    if use_all_joints_on_each_bp:
        body_parts = config.body_parts_entire_body if not visibility else config.body_parts_invis_entire_body
    else:
        body_parts = config.body_parts if not visibility else config.body_parts_invis

    bps = list(body_parts._fields)

    cand_framesPerBP, metric, querys = ntu_similarity_prepart(net, x, f, y, body_parts, bps, dist, slide, window_size)

    querys_bp_flatten = []
    for query in querys:
        query_bp_flatten = [np.concatenate([bp[p_idx] for bp in query]) for p_idx in range(len(query[0]))]
        querys_bp_flatten.append(query_bp_flatten)
    c_bp_flatten = [np.concatenate([bp[p_idx] for bp in cand_framesPerBP]) for p_idx in range(len(cand_framesPerBP[0]))]
    paths = [dtw_path(np.array(query_flatten), np.array(c_bp_flatten))[0] for query_flatten in querys_bp_flatten]

    finals = [[] for _ in range(4)]
    for idx, (query, path) in enumerate(zip(querys, paths)):
        for path_idx in range(len(path)):
            sims = []
            for bp in range(len(bps)):
                if dist == "cosine":
                    metric_sim = metric(torch.Tensor(query[bp][path[path_idx][0]]),
                                        torch.Tensor(cand_framesPerBP[bp][path[path_idx][1]])).numpy()
                else:
                    metric_sim = metric(torch.Tensor(query[bp][path[path_idx][0]]).unsqueeze(0),
                                        torch.Tensor(cand_framesPerBP[bp][path[path_idx][1]]).unsqueeze(0)).numpy()[0]
                sims.append(metric_sim)
            sims = np.sort(np.array(sims), axis=None)
            sims = sims[:min_num]
            finals[idx].append(np.mean(sims))

    # origin, flipped, arm_flipped, leg_flipped
    final_sim_calculated = [np.mean(final_v) for final_v in finals]  # different paths don't `np.mean(_, axis=1)`

    if dist == 'cosine':
        return tuple(final_sim_calculated)
    else:
        return tuple([1 / item for item in final_sim_calculated])


def load_model(config, path):
    n = networks_bpe.AutoEncoder_bpe(config)
    # n.load_state_dict(torch.load(path))
    n.load_state_dict(load_ckpt_from_path(path))
    n.to(config.device)
    n.eval()

    return n


def load_ckpt_from_path(model_path: str) -> OrderedDict:
    assert os.path.exists(model_path)
    print('Loading model from {}'.format(model_path))
    state_dict = torch.load(model_path)

    return state_dict


def print_amt_corr(models, all_data_list):
    for model, target_data in zip(models, all_data_list):
        print(f'MODEL {model.split("/")[-1]}', end="\t")
        print(calc_amt_corr(target_data))


def calc_amt_corr(target_data):
    base1 = target_data
    target_data['body'] = target_data[["bpe_ori_sim", "bpe_flipped_sim"]].max(axis=1)
    target_data['parts'] = target_data[
        ["bpe_ori_sim", "bpe_flipped_sim", "bpe_arm_flipped_sim", "bpe_leg_flipped_sim"]].max(axis=1)
    bpe_none = bpe_correlation(base1, target_data, 'AMT_score', 'bpe_ori_sim')
    bpe_body = bpe_correlation(base1, target_data, 'AMT_score', 'body')
    bpe_parts = bpe_correlation(base1, target_data, 'AMT_score', 'parts')
    return bpe_none, bpe_body, bpe_parts


def bpe_correlation(df1, df2, amt_score_column_name, df_column_name2, rank=True):
    df11 = df1.copy()
    df = df2.copy()

    if rank:
        return df11[amt_score_column_name].corr(df[df_column_name2], method='spearman')
        # return df[amt_score_column_name].corr(df[df_column_name2], method='kendall')
    else:
        return df[amt_score_column_name].corr(df[df_column_name2])


def training_set_mean_height(mean_pose_bpe):
    if mean_pose_bpe.shape[0] == 19:
        r_ankle_coord_y = mean_pose_bpe[8, 1] + mean_pose_bpe[16, 1]  # right ankle coord. centered by middle hip
        l_ankle_coord_y = mean_pose_bpe[11, 1] + mean_pose_bpe[17, 1]  # left ankle coord. centered by middle hip
        avg_ankle_coord_y = (r_ankle_coord_y + l_ankle_coord_y) / 2
        nose_coord_y = mean_pose_bpe[12, 1]
        mean_height = avg_ankle_coord_y - nose_coord_y
    else:
        torso_ys = mean_pose_bpe[-16:-1, 1]
        r_ankle_coord_y, l_ankle_coord_y = torso_ys[11], torso_ys[14]
        avg_ankle_coord_y = (r_ankle_coord_y + l_ankle_coord_y) / 2
        nose_coord_y = torso_ys[0]
        mean_height = avg_ankle_coord_y - nose_coord_y

    return mean_height


def start_ntu_similarity(epoch_path, args, all_data, scale,
                         mean_pose_bpe, std_pose_bpe, mean_height):
    config = Config(args)
    net = load_model(config, epoch_path)

    loaded_items = {}
    similarities = []
    for row in tqdm(all_data.index, desc="processing each data row"):
        query = all_data['sample1'][row]
        candidate = all_data['sample2'][row]

        query_action_idx = query[-3:]
        query_json_path = os.path.join(args.ntu_dir, query_action_idx, query + '.json')
        query_motion = cocopose2motion(config.unique_nr_joints, query_json_path, scale=scale,
                                       visibility=config.invisibility_augmentation, mean_height=mean_height)
        
        if query in list(loaded_items.keys()):
            query_tensor = loaded_items[query]
            flipped_query = loaded_items[query + '_flipped']
        else:
            query_tensor, flipped_query = preprocess_motion2tensor(config, query_motion, mean_pose_bpe,
                                                                   std_pose_bpe,
                                                                   flip=args.use_flipped_motion,
                                                                   visibility=config.invisibility_augmentation,
                                                                   use_all_joints_on_each_bp=config.use_all_joints_on_each_bp)
            loaded_items[query] = query_tensor
            loaded_items[query + '_flipped'] = flipped_query

        candidate_action_idx = candidate[-3:]
        candidate_json_path = os.path.join(args.ntu_dir, candidate_action_idx,
                                           candidate + '.json')
        candidate_motion = cocopose2motion(config.unique_nr_joints, candidate_json_path, scale=scale,
                                           visibility=config.invisibility_augmentation, mean_height=mean_height)

        if candidate in list(loaded_items.keys()):
            cand_tensor = loaded_items[candidate]
        else:
            cand_tensor, flipped_cand = preprocess_motion2tensor(config, candidate_motion, mean_pose_bpe,
                                                                 std_pose_bpe,
                                                                 flip=args.use_flipped_motion,
                                                                 visibility=config.invisibility_augmentation,
                                                                 use_all_joints_on_each_bp=config.use_all_joints_on_each_bp)
            loaded_items[candidate] = cand_tensor
            loaded_items[candidate + '_flipped'] = flipped_cand

        if args.use_global_dtw:
            similarity_calculated = ntu_similarity_global_dtw(net, config, query_tensor,
                                                              flipped_query,
                                                              cand_tensor,
                                                              window_size=32,
                                                              slide=2,
                                                              visibility=config.invisibility_augmentation,
                                                              dist=args.similarity_distance_metric,
                                                              use_all_joints_on_each_bp=config.use_all_joints_on_each_bp)
        else:
            similarity_calculated = ntu_similarity(net, config, query_tensor,
                                                   flipped_query, cand_tensor,
                                                   window_size=32, slide=2,
                                                   visibility=config.invisibility_augmentation,
                                                   dist=args.similarity_distance_metric,
                                                   use_all_joints_on_each_bp=config.use_all_joints_on_each_bp)

        similarities.append(similarity_calculated)

    return similarities


def basic_configuration():
    base_csv = os.path.join(os.path.dirname(__file__), "NTU_motion_similarity_AMT_final_200204.csv")

    all_data = pd.read_csv(base_csv, index_col=0)
    if args.limit > 0:
        all_data = all_data[:args.limit]

    mean_pose_bpe = np.load(config.meanpose_rc_path)
    std_pose_bpe = np.load(config.stdpose_rc_path)

    if args.use_trainingset_mean_height:
        mean_height = training_set_mean_height(mean_pose_bpe)
    else:
        mean_height = None

    return all_data, mean_pose_bpe, std_pose_bpe, mean_height


def main():
    ntu_dir = args.ntu_dir

    # expereiments settings
    img_size = [480, 854]  # height, width

    save_csv_dir = "results"
    save_csv_name_base = "AMT_for_corr_"
    if not os.path.exists(os.path.join(ntu_dir, save_csv_dir)):
        os.makedirs(os.path.join(ntu_dir, save_csv_dir))


    ######################################################################################################
    h, w, scale = pad_to_height(img_size[0], 1080, 1920)

    all_data, mean_pose_bpe, std_pose_bpe, mean_height = basic_configuration()

    models_path = [args.model_path]

    if args.use_mp:
        with mp.Pool(min(15, mp.cpu_count() - 1)) as p:
            nets_similarities = [p.apply_async(start_ntu_similarity, args=(epoch_path, args, all_data, scale,
                                                                           mean_pose_bpe, std_pose_bpe, mean_height))
                                 for epoch_path in models_path]
            nets_similarities = [result.get() for result in nets_similarities]
    else:
        nets_similarities = []
        for epoch_path in models_path:
            net_similarity = start_ntu_similarity(epoch_path, args, all_data, scale,
                                                  mean_pose_bpe, std_pose_bpe, mean_height)
            nets_similarities.append(net_similarity)

    if args.print_amt_corr:
        all_data_epoch_order = []

    for net_idx in range(len(models_path)):
        csv_name = save_csv_name_base + str(models_path[net_idx]).split("/")[-3] + "_" + str(models_path[net_idx].split("/")[-1]) + ".csv"
        out_csv = os.path.join(ntu_dir, save_csv_dir, csv_name)

        similarity = np.asarray(nets_similarities[net_idx])
        origin = similarity[:, 0].tolist()
        flipped = similarity[:, 1].tolist()
        arm_flipped = similarity[:, 2].tolist()
        leg_flipped = similarity[:, 3].tolist()
        all_data_copy = all_data.copy()
        all_data_copy['bpe_ori_sim'] = origin
        all_data_copy['bpe_flipped_sim'] = flipped
        all_data_copy['bpe_arm_flipped_sim'] = arm_flipped
        all_data_copy['bpe_leg_flipped_sim'] = leg_flipped
        all_data_copy.to_csv(out_csv)

        if args.print_amt_corr:
            all_data_epoch_order.append(all_data_copy)

    if args.print_amt_corr:
        print_amt_corr(models_path, all_data_epoch_order)


class WorkerManager:
    worker_filename = "last_worker_epoch"

    @staticmethod
    def get_last_worker_path(model_path):
        return os.path.join(model_path, WorkerManager.worker_filename + "_" + str(args.limit) + "_dtw" + str(args.use_global_dtw))

    @staticmethod
    def get_last_worker_epoch(model_path):
        last_worker_path = WorkerManager.get_last_worker_path(model_path)

        last_epoch = 1

        # if no last workerr written
        if not os.path.isfile(last_worker_path):
            return last_epoch

        try:
            with open(last_worker_path, "r") as fr:
                written_epoch = int(fr.read().strip())
            if written_epoch > 0:
                last_epoch = written_epoch + 1
        except Exception as e:
            # invalid file
            pass

        return last_epoch

    @staticmethod
    def update_last_worker_epoch(model_path, epoch):
        last_worker_path = WorkerManager.get_last_worker_path(model_path)

        with open(last_worker_path, "w") as fw:
            fw.write(str(epoch))


def start_worker():
    all_data, mean_pose_bpe, std_pose_bpe, mean_height = basic_configuration()

    if args.limit == -1:
        args.limit = "all"

    img_size = [480, 854]  # height, width
    h, w, scale = pad_to_height(img_size[0], 1080, 1920)

    # worker write on logdir
    log_dir = os.path.join(os.path.dirname(args.model_path), "log")
    summary_writer = SummaryWriter(os.path.join(log_dir, 'train.events'))

    same_counter = 0

    # wait for 10 min
    while True:
        # Check model dir's last written epoch
        epoch = WorkerManager.get_last_worker_epoch(args.model_path)

        modelname = 'model_epoch' + str(epoch) + '.pth'
        epoch_path = os.path.join(args.model_path, modelname)

        if not os.path.isfile(epoch_path):
            same_counter += 1
            time.sleep(60 * 8)
            if same_counter > 3:
                break
            continue
        else:
            same_counter = 0

        net_similarity = start_ntu_similarity(epoch_path, args, all_data, scale,
                                              mean_pose_bpe, std_pose_bpe, mean_height)
        similarity = np.asarray(net_similarity)

        origin = similarity[:, 0].tolist()
        flipped = similarity[:, 1].tolist()
        arm_flipped = similarity[:, 2].tolist()
        leg_flipped = similarity[:, 3].tolist()

        all_data_copy = all_data.copy()
        all_data_copy['bpe_ori_sim'] = origin
        all_data_copy['bpe_flipped_sim'] = flipped
        all_data_copy['bpe_arm_flipped_sim'] = arm_flipped
        all_data_copy['bpe_leg_flipped_sim'] = leg_flipped

        bpe_none, bpe_body, bpe_parts = calc_amt_corr(all_data_copy)

        # write tensorboard
        summary_writer.add_scalar(f"similarity/0_none_{args.limit}_dtw{args.use_global_dtw}", bpe_none, global_step=epoch)
        summary_writer.add_scalar(f"similarity/1_body_{args.limit}_dtw{args.use_global_dtw}", bpe_body, global_step=epoch)
        summary_writer.add_scalar(f"similarity/2_parts_{args.limit}_dtw{args.use_global_dtw}", bpe_parts, global_step=epoch)

        # write epoch num on txt
        WorkerManager.update_last_worker_epoch(args.model_path, epoch=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test")
    parser.add_argument('--data_dir', default="./bpe-datasets/SARA_released/", help="path to dataset dir")
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('--ntu_dir', default="./bpe-datasets/NTU_joints", help="processed NTU pose dir")

    parser.add_argument('-g', '--gpu_ids', type=int, default=2, required=False)
    parser.add_argument('--similarity_distance_metric', choices=["cosine", "l2"], default="cosine")

    parser.add_argument('--use_flipped_motion', action='store_true', default=True,
                        help="whether to use one decoder per one body part")  # TODO: not implemented with False
    parser.add_argument('--use_trainingset_mean_height', action='store_true', default=True)
    parser.add_argument('--use_invisibility_aug', action='store_true')
    parser.add_argument('--use_global_dtw', action='store_true')
    parser.add_argument('--use_all_joints_on_each_bp', action='store_true')

    # DEBUG - early evaluation
    parser.add_argument('--limit', type=int, default=-1, help="only use partial to get similarity")
    parser.add_argument('--print_amt_corr', action='store_true', default=True)
    parser.add_argument('--use_mp', action='store_true', default=False)

    # realtime worker mode
    parser.add_argument('--worker', action='store_true', default=False)

    args = parser.parse_args()

    if args.use_mp:
        mp.set_start_method('spawn', force=True)

    config = Config(args)

    if args.worker:
        start_worker()
    else:
        main()
