from __future__ import print_function

import torch
import torch.nn as nn


class QuadrupletLoss:
    def __init__(self, config):
        self.margin = config.quadruplet_margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin)
        self.alpha = config.variation_control_param
        self.sim_loss_weight = config.quadruplet_sim_weight
        # distance metric of triplet loss
        self.triplet_distance = config.triplet_distance
        # distance metric of similarity loss
        self.similarity_distance_metric = config.similarity_distance_metric
        assert self.triplet_distance in ['cosine', 'l2', None]
        assert self.similarity_distance_metric in ['cosine', 'l2', None]

        self.dist = self.cosine_distance if self.similarity_distance_metric == "cosine" else self.l2_distance
        
    def triplet_cosine_distance(self, anchor, p, n):
        distance_p = self.cosine_distance(anchor, p)
        distance_n = self.cosine_distance(anchor, n)
        hinge = torch.clamp(self.margin +  distance_p - distance_n, min=0.0)
        loss = torch.mean(hinge)
        return loss

    def cosine_distance(self, p, sp, dim=1, eps=1e-5):
        cosine_sim = nn.CosineSimilarity(dim=dim, eps=eps)
        return (1 - cosine_sim(p, sp)) / 2

    def l2_distance(self, p, sp, norm=2):
        l2_dist = nn.PairwiseDistance(p=norm)
        return l2_dist(p, sp)

    def qloss(self, anchor, p, sp, n, variation_score):
        anchor = anchor.detach()

        if self.triplet_distance == "cosine":
            triplet_loss = self.triplet_cosine_distance(anchor, p, n) + self.triplet_cosine_distance(anchor, sp, n)
        else: # the case of L2 distance
            triplet_loss = self.triplet_loss(anchor, p, n) + self.triplet_loss(anchor, sp, n)

        var = self.alpha * variation_score.unsqueeze(1).repeat(1, anchor.shape[-1])  # Target distance between p & sp

        sim_loss = torch.mean(torch.pow(self.dist(p, sp) - var, 2))  # L2 loss
        # sim_loss = torch.mean(torch.abs(self.dist(p, sp) - var))  # L1 loss
        loss = triplet_loss + self.sim_loss_weight * sim_loss

        return loss

    def tloss(self, anchor, p, sp, n):
        triplet_loss = self.triplet_loss(anchor, p, n) + self.triplet_loss(anchor, sp, n)
        return triplet_loss

    def sloss(self, p, sp, variation_score):
        var = self.alpha * variation_score
        sim_loss = torch.mean(torch.pow(self.dist(p, sp) - var, 2))
        return sim_loss
