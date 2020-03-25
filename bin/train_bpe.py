import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bpe import Config
from bpe.agent import agents_bpe
from bpe.dataset.datasets_bpe import SARADataset
from bpe.functional.utils import cycle, move_to_device
from bpe.model import networks_bpe

torch.backends.cudnn.benchmark = True


def get_markdown_table(dict_data=None):
    hps = ['| key | value |', '| --- | --- |']
    if dict_data:
        for key, value in dict_data.items():
            value_str = str(value).replace('\n', '<br/>')
            hps.append(f'| {key} | {value_str} |')

    return "\r\n".join(hps)


def add_hps_using(config, train_tb):
    white_list = set(["triplet_distance", "similarity_distance_metric", "dataset"])

    config_all_dict = {k: v for k, v in Config.__dict__.items()}
    config_all_dict.update(config.__dict__.copy())
    config_logging_dict = {k: v for k, v in config_all_dict.items() if type(v) in [int, float, bool] or k in white_list}
    train_tb.add_text('hyperparams', get_markdown_table(config_logging_dict))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='bpe', help='Experiment name')
    parser.add_argument('-g', '--gpu_ids', type=str, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--dataset', choices=["unity", "mixamo"], default="unity",
                        help="whether to use one decoder per one body part")
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")

    # Experiments argumen ts
    parser.add_argument('--use_footvel_loss', action='store_true', help="use footvel loss")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--use_all_joints_on_each_bp', action='store_true',
                        help="using all joints on each body part as input, as opposed to particular body part")
    parser.add_argument('--triplet_distance', choices=["cosine", "l2"], default=None)
    parser.add_argument('--similarity_distance_metric', choices=["cosine", "l2"], default="cosine")
    parser.add_argument('--sim_loss_weight', type=float, default=None)

    parser.add_argument('--norecon', action='store_true')

    parser.add_argument('--logdir', type=str, default=None, help="change model/logdir")
    args = parser.parse_args()

    config = Config(args)

    # create the network
    net = networks_bpe.AutoEncoder_bpe(config)
    # print(net)
    net = torch.nn.DataParallel(net)
    net.to(config.device)

    # create tensorboard writer
    summary_writer = SummaryWriter(os.path.join(config.log_dir, 'train.events'))
    add_hps_using(config, summary_writer)

    # create dataloader
    train_dataset = SARADataset('train', config)
    val_dataset = SARADataset('test', config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
                              worker_init_fn=lambda _: np.random.seed(), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
                            worker_init_fn=lambda _: np.random.seed(), pin_memory=True)

    # validation is performed in the middle of training epoch
    # as a single step, rather then a full val data pass
    val_loader = cycle(val_loader)

    # create training agent
    tr_agent = agents_bpe.Agent3x_bpe(config, net)
    clock = tr_agent.clock

    summary_writer.add_scalar('learning_rate', config.lr, 0)

    min_val_loss = np.inf

    # start training
    for e in range(config.nr_epochs):
        
        epoch_val_loss = []

        # begin iteration
        pbar = tqdm(train_loader)

        for b, data_input in enumerate(pbar):
            # training
            # move data to appropriate device
            data_input = move_to_device(data_input, config.device, non_blocking=True)

            # train step
            losses = tr_agent.train_func(data_input)

            losses_values = {k: v.item() for k, v in losses.items()}

            # record loss to tensorboard
            for k, v in losses_values.items():
                summary_writer.add_scalar("train/" + k, v, clock.step)
            summary_writer.add_scalar("train/total_loss", sum(losses_values.values()), clock.step)

            pbar.set_description("EPOCH[{}][{}/{}]".format(e, b, len(train_loader)))

            # validation step
            if clock.step % config.val_frequency == 0:
                data_input_val = next(val_loader)
                # move data to appropriate device
                data_input_val = move_to_device(data_input_val, config.device)

                losses = tr_agent.val_func(data_input_val)

                losses_values = {k: v.item() for k, v in losses.items()}

                for k, v in losses_values.items():
                    summary_writer.add_scalar("valid/" + k, v, clock.step)
                summary_writer.add_scalar("valid/total_loss", sum(losses_values.values()), clock.step)
                epoch_val_loss.append(sum(losses_values.values()))

            if clock.lr_minibatch >= (len(pbar) // config.lr_update_frequency_per_epoch) - 1:
                clock.lr_step_update()
                tr_agent.update_learning_rate()
                clock.lr_minibatch = 0
                summary_writer.add_scalar('learning_rate', tr_agent.optimizer.param_groups[-1]['lr'], clock.step + 1)

            clock.tick()

        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_network()
        tr_agent.save_network('latest.pth.tar')
        
        mean_epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
        if min_val_loss > mean_epoch_val_loss:
            print("saving model model_best.pth.tar")
            tr_agent.save_network('model_best.pth.tar')
            min_val_loss = mean_epoch_val_loss

        clock.tock()

    # close tensorboard writers
    if summary_writer is not None:
        summary_writer.close()


if __name__ == '__main__':
    main()
