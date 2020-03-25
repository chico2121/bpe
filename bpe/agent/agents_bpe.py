import torch

from bpe.agent.base_agent import BaseAgent
from bpe.model.quadruplet_loss import QuadrupletLoss


def _foot_loss_base(leg, foot_idx):
    loss = leg[:, foot_idx, 1:] - leg[:, foot_idx, :-1]
    return loss


class Agent3x_bpe(BaseAgent):
    def __init__(self, config, net):
        super(Agent3x_bpe, self).__init__(config, net)

        self.quadruplet_loss = QuadrupletLoss(config)
        self.quadruplet_weight = config.quadruplet_weight
        self.recon_weight = config.recon_weight
        self.num_of_body_parts = config.nr_body_parts
        self.body_parts = config.body_parts
        self.body_part_names = config.body_part_names
        self.use_all_joints_on_each_bp = config.use_all_joints_on_each_bp
        self.motion_embedding_l2reg = config.motion_embedding_l2reg

    def forward(self, data):
        # update loss metric
        losses = {}

        # obtain encoded and decoded vectors
        outputs, motion_vecs, body_vecs, view_vecs = self.net.forward(data)
        variation_score = data['variation_p_sp']

        # body parts motion loss
        losses['m_quad1'] = self.quadruplet_weight * torch.mean(torch.stack(
            [self.quadruplet_loss.qloss(motion_vecs[3][j], motion_vecs[0][j], motion_vecs[4][j], motion_vecs[2][j],
                                        variation_score) for j in range(len(motion_vecs[3]))]))
        losses['m_quad2'] = self.quadruplet_weight * torch.mean(torch.stack(
            [self.quadruplet_loss.qloss(motion_vecs[4][j], motion_vecs[1][j], motion_vecs[3][j], motion_vecs[2][j],
                                        variation_score) for j in range(len(motion_vecs[4]))]))
        losses['m_quad3'] = self.quadruplet_weight * torch.mean(torch.stack(
            [self.quadruplet_loss.qloss(motion_vecs[1][j], motion_vecs[4][j], motion_vecs[0][j], motion_vecs[2][j],
                                        variation_score) for j in range(len(motion_vecs[1]))]))
        losses['m_quad4'] = self.quadruplet_weight * torch.mean(torch.stack(
            [self.quadruplet_loss.qloss(motion_vecs[0][j], motion_vecs[3][j], motion_vecs[1][j], motion_vecs[2][j],
                                        variation_score) for j in range(len(motion_vecs[0]))]))

        # L2 regularization on motion embeddings
        if self.motion_embedding_l2reg:
            losses['m_l2reg'] = 1e-2 * torch.mean(torch.stack(
                [(motion_vecs[i][j].norm(p=2, dim=1) - 1.0).pow(2) for i in range(len(motion_vecs)) for j in
                 range(len(motion_vecs[0]))]))

        # body parts loss
        if self.use_all_joints_on_each_bp:
            # because body encoder use entire body instead of body parts joints, and in terms of the whole body, there is little difference between p and sp.
            # so, semi-positve's embedding is not considered.
            losses['b_tpl1'] = self.triplet_weight * self.triplet_loss(body_vecs[2], body_vecs[0], body_vecs[1])
            losses['b_tpl2'] = self.triplet_weight * self.triplet_loss(body_vecs[3], body_vecs[1], body_vecs[0])
        else:
            losses['b_tpl1'] = self.triplet_weight * torch.mean(torch.stack(
                [self.triplet_loss(body_vecs[2][j], body_vecs[0][j], body_vecs[1][j]) for j in range(len(body_vecs[2]))]))
            losses['b_tpl2'] = self.triplet_weight * torch.mean(torch.stack(
                [self.triplet_loss(body_vecs[3][j], body_vecs[0][j], body_vecs[4][j]) for j in range(len(body_vecs[3]))]))
            losses['b_tpl3'] = self.triplet_weight * torch.mean(torch.stack(
                [self.triplet_loss(body_vecs[5][j], body_vecs[1][j], body_vecs[0][j]) for j in range(len(body_vecs[5]))]))
            losses['b_tpl4'] = self.triplet_weight * torch.mean(torch.stack(
                [self.triplet_loss(body_vecs[6][j], body_vecs[1][j], body_vecs[7][j]) for j in range(len(body_vecs[6]))]))

        # view loss
        losses['v_tpl1'] = self.triplet_weight * self.triplet_loss(view_vecs[3], view_vecs[0], view_vecs[1])
        losses['v_tpl2'] = self.triplet_weight * self.triplet_loss(view_vecs[2], view_vecs[1], view_vecs[0])

        # body part reconstruction loss
        body_parts_targets = self._get_target(data)
        for i, bp in enumerate(body_parts_targets):
            # calculate mse as a stacked tensor instead of loop
            mean_bp_loss = self.recon_weight * self.mse(torch.stack(outputs[i]), torch.stack(bp))
            losses['rec_' + self.body_part_names[i]] = mean_bp_loss

        # foot velocity loss
        if self.use_footvel_loss:
            foot_vel_outputs = self._get_foot_vel(outputs, self.foot_idx)
            foot_vel_targets = self._get_foot_vel(body_parts_targets, self.foot_idx)
            losses['foot_vel'] = self.footvel_loss_weight * self.mse(foot_vel_outputs, foot_vel_targets)

        return losses

    def _get_foot_vel_for_md(self, outputs, foot_idx):
        result = []  # result: (8, # of batch, len(foot_idx)=2, # of frames -1)
        if len(outputs) == len(self.targets_name):
            for out1 in outputs:
                torse_loss = out1[4][:, -2:, 1:]
                rl = _foot_loss_base(out1[2], foot_idx) + torse_loss
                ll = _foot_loss_base(out1[3], foot_idx) + torse_loss
                result.append(rl + ll)
        else:
            for i in range(len(self.targets_name)):
                torse_loss = outputs[4][i][:, -2:, 1:]
                rl = _foot_loss_base(outputs[2], foot_idx) + torse_loss
                ll = _foot_loss_base(outputs[3], foot_idx) + torse_loss
                result.append(rl + ll)
        return torch.stack(result)

    def _get_target(self, data):
        target = []
        for bp in self.body_part_names:
            assert bp in data.keys()
            # [p11, p12, p21, p22, sp11, sp12, sp21, sp22, n11, n12, n21, n22]
            bp_target = [
                data[bp][:, 0, 0, 0, :, :],
                data[bp][:, 0, 0, 1, :, :],
                data[bp][:, 0, 1, 0, :, :],
                data[bp][:, 0, 1, 1, :, :],
                data[bp][:, 1, 0, 0, :, :],
                data[bp][:, 1, 0, 1, :, :],
                data[bp][:, 1, 1, 0, :, :],
                data[bp][:, 1, 1, 1, :, :],
                data[bp][:, 2, 0, 0, :, :],
                data[bp][:, 2, 0, 1, :, :],
                data[bp][:, 2, 1, 0, :, :],
                data[bp][:, 2, 1, 1, :, :]
            ]
            target.append(bp_target)
        return target

    def _get_foot_vel(self, outputs, foot_idx):
        result = []
        rl = outputs[2]
        ll = outputs[3]
        torso = outputs[4]
        assert len(rl) == len(ll)

        for comb_idx in range(len(rl)):
            torse_loss = torso[comb_idx][:, -2:, 1:]
            rl_loss = _foot_loss_base(rl[comb_idx], foot_idx) + torse_loss
            ll_loss = _foot_loss_base(ll[comb_idx], foot_idx) + torse_loss
            result.append(rl_loss + ll_loss)
        return torch.stack(result)
