import os
import math
from collections import OrderedDict
import itertools

import torch
from tslearn.metrics import dtw_path
import numpy as np

from bpe.model import networks_bpe


class SimilarityAnalyzer:
    def __init__(self, config, model_path):
        self.body_parts = config.default_body_parts
        self.body_parts_name = config.body_part_names

        # get similarity model
        self.motion_encoders = self._get_motion_model(config, model_path)
        self.cosine_score = torch.nn.CosineSimilarity(dim=0, eps=1e-50)

    def _get_motion_model(self, config, model_path):
        # define network
        print('Building model')
        network = networks_bpe.AutoEncoder_bpe(config)
        # load pretrained model
        network.load_state_dict(self.load_ckpt_from_path(model_path, device=config.device))
        # extract only motion encoders
        network = network.mot_encoders
        # move to appropriate device
        network.to(config.device)
        network.eval()
        print('Model is ready')
        return network

    @staticmethod
    def load_ckpt_from_path(model_path: str, device: str = "gpu") -> OrderedDict:
        assert os.path.exists(model_path)
        print('Loading model from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=device)

        # TODO: Only for Old ckpts. Deprecate later
        if 'module.' in state_dict.__iter__().__next__():
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        return state_dict

    def oversample_sequences(self, seq, window_size, stride):
        out_seq = []
        for i in range(math.ceil(((seq.shape[-1] - window_size + 1) / stride))):
            out_seq.append(seq[:, :, i * stride: i * stride + window_size])
        return out_seq

    def _chunked_iterable(self, iterable, size):
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, size))
            if len(chunk) != size:
                break
            yield chunk

    def get_motion_encodings_by_body_part_for_sequences(self, seq):
        encodings_sequences = []
        for subseq in seq:
            encodings_bp = []
            for bp_idx in range(len(self.body_parts_name)):
                current_bp_seq = subseq[:, self.body_parts[bp_idx], :]
                current_bp_encoding = self.motion_encoders[bp_idx](current_bp_seq)[0].reshape(-1)
                encodings_bp.append(current_bp_encoding.cpu().detach().numpy())
            encodings_sequences.append(encodings_bp)
        return encodings_sequences

    def _get_similarity(self, seq1_features, seq2_features):
        path, dist = dtw_path(np.array(seq1_features), np.array(seq2_features))
        similarities_per_path = []
        for i in range(len(path)):
            cosine_sim = self.cosine_score(torch.Tensor(seq1_features[path[i][0]]),
                                           torch.Tensor(seq2_features[path[i][1]])).numpy()
            similarities_per_path.append(cosine_sim)
        total_path_similarity = sum(similarities_per_path) / len(path)
        return total_path_similarity

    def get_embeddings(self, seq, video_window_size, video_stride):
        # extract sub-sequences defined by window size and stride
        seq = self.oversample_sequences(seq, video_window_size, video_stride)
        # extract motion embeddings
        seq_features = self.get_motion_encodings_by_body_part_for_sequences(seq)
        return seq_features

    def get_similarity_score(self, seq1_features, seq2_features, similarity_window_size):

        similarity_score_per_window = []

        for subseq1_features, subseq2_features in zip(*(self._chunked_iterable(seq1_features, similarity_window_size),
                                                        self._chunked_iterable(seq2_features, similarity_window_size))):
            assert len(subseq1_features) == len(subseq2_features) and len(subseq2_features) != 0

            similarity_score_by_body_part = {}
            for bp_idx, bp in enumerate(self.body_parts_name):
                subseq1_features_bp = [subseq1_features[subseq_temporal_idx][bp_idx] for subseq_temporal_idx in
                                       range(len(subseq1_features))]
                subseq2_features_bp = [subseq2_features[subseq_temporal_idx][bp_idx] for subseq_temporal_idx in
                                       range(len(subseq2_features))]
                similarity_score_by_body_part[bp] = self._get_similarity(subseq1_features_bp, subseq2_features_bp)

            similarity_score_per_window.append(similarity_score_by_body_part)

        return similarity_score_per_window
