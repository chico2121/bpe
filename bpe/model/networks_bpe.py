import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None, compress=False, batchnorm=False):
        super(Encoder, self).__init__()

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 2 if compress else len(channels) - 1

        for i in range(nr_layer):
            if convpool is None:
                pad = (kernel_size - 1) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=2))
                if batchnorm:
                    model.append(nn.BatchNorm1d(channels[i + 1]))
                model.append(acti)
            else:  # body & view
                pad = (kernel_size - 1) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=1))
                if batchnorm:
                    model.append(nn.BatchNorm1d(channels[i + 1]))
                model.append(acti)
                model.append(convpool(kernel_size=2, stride=2))  # nn.MaxPool1d

        self.global_pool = global_pool
        self.compress = compress

        self.model = nn.Sequential(*model)

        if self.compress:
            self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)

        self.last_conv = nn.Conv1d(channels[-1], channels[-1], kernel_size=1, bias=False)

    def forward(self, x):
        x = self.model(x)
        if self.global_pool is not None:
            ks = x.shape[-1]
            x = self.global_pool(x, ks)  # F.max_pool1d
            if self.compress:
                x = self.conv1x1(x)
        else:
            x = self.last_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(Decoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)

        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)  # whether to add tanh a last?

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class AutoEncoder_bpe(nn.Module):
    def __init__(self, config):
        super(AutoEncoder_bpe, self).__init__()

        self.mot_en_channels = config.mot_en_channels
        self.body_en_channels = config.body_en_channels
        self.view_en_channels = config.view_en_channels
        self.de_channels = config.de_channels
        self.body_part_names = config.body_part_names
        self.invisibility_augmentation = config.invisibility_augmentation
        self.use_all_joints_on_each_bp = config.use_all_joints_on_each_bp

        self.view_encoder = Encoder(self.view_en_channels, kernel_size=7, global_pool=F.avg_pool1d,
                                    convpool=nn.AvgPool1d, compress=True, batchnorm=config.Batchnorm)
        self.mot_encoders = self._get_mot_encoders(config)
        
        # if you use use_all_joints_on_each_bp option, motion & body encoder are fed the entire body joints, not the body part joints.
        if self.use_all_joints_on_each_bp:
            self.body_encoder = Encoder(self.body_en_channels, kernel_size=7, global_pool=F.max_pool1d,
                                         convpool=nn.MaxPool1d, compress=True, batchnorm=config.Batchnorm)
        else:
            self.body_encoders = self._get_body_encoders(config)
        self.decoders = self._get_decoders(config)

    def _get_mot_encoders(self, config):
        """
        :return: list of encoders in order
            [ra_mot_encoder, la_mot_encoder, rl_mot_encoder, ll_mot_encoder, torso_mot_encoder]
        """
        mot_encoders = []
        for i in range(len(self.mot_en_channels)):
            cur_encoder = Encoder(self.mot_en_channels[i], kernel_size=7, batchnorm=config.Batchnorm)
            mot_encoders.append(cur_encoder)
        return nn.ModuleList(mot_encoders)

    def _get_body_encoders(self, config):
        """
        :param config:
        :return: list of encoders in order
            [ra_body_encoder, la_body_encoder, rl_body_encoder, ll_body_encoder, torso_body_encoder]
        """
        body_encoders = []
        for i in range(len(self.body_en_channels)):
            cur_encoder = Encoder(self.body_en_channels[i], kernel_size=7, global_pool=F.max_pool1d,
                                  convpool=nn.MaxPool1d, compress=True, batchnorm=config.Batchnorm)
            body_encoders.append(cur_encoder)
        return nn.ModuleList(body_encoders)

    def _get_decoders(self, config):
        """
        :return: list of decoders in order
            [ra_decoder, la_decoder, rl_decoder, ll_decoder, torso_decoder # with velocity([-2,-1])]
        """
        decoders = []
        for i in range(len(self.de_channels)):
            cur_decoder = Decoder(self.de_channels[i])
            decoders.append(cur_decoder)
        return nn.ModuleList(decoders)

    def cross_with_quadruplet(self, inputs):
        invis = '_invis' if self.invisibility_augmentation else ''
        bp_inputs = [inputs[bp + invis] for bp in self.body_part_names]

        # feed-forward inputs through encoders
        m_p, m_sp, m_n, m_p22, m_sp22 = self._mot_encoders_forward(bp_inputs)
        b1, b2, b_n12, b_sp12, b_sp22, b_p21, b_sp21, b_sp11 = self._body_encoders_forward(bp_inputs)
        v1, v2, v_n21, v_p12 = self._view_encoders_forward(inputs)

        # combine encoded parts
        motion_ebds = [m_p, m_sp, m_n]
        body_ebds = [b1, b2]
        view_ebds = [v1, v2]

        # get decoded outputss
        outputs = []  # [p11, p12, p21, p22, sp11, sp12, sp21, sp22, n11, n12, n21, n22]
        for mot in motion_ebds:
            for bd in body_ebds:
                for v in view_ebds:
                    concat_ebd = self._concat_bpe(mot, bd, v)
                    output = [d(concat_ebd[k]) for k, d in enumerate(self.decoders)]
                    outputs.append(output)

        # transpose func : final shape --> [bps, combinations, # of batches, # of joints, # of frames]
        outputs = list(map(list, zip(*outputs)))
        motion_vecs = [m_p, m_sp, m_n, m_p22, m_sp22]
        body_vecs = self._reshape_encoded_inputs([b1, b2, b_n12, b_sp12, b_sp22, b_p21, b_sp21, b_sp11])
        view_vecs = [v1.reshape(v1.shape[0], -1), v2.reshape(v2.shape[0], -1),
                     v_n21.reshape(v_n21.shape[0], -1), v_p12.reshape(v_p12.shape[0], -1)]

        return outputs, motion_vecs, body_vecs, view_vecs
    
    def cross_with_all_joints_on_each_bp_quadruplet(self, inputs):
        all_joints = '_all_joints' if self.use_all_joints_on_each_bp else ''
        invis = '_invis' if self.invisibility_augmentation else ''
        bp_inputs = [inputs[bp + all_joints + invis] for bp in self.body_part_names]

        # feed-forward inputs through encoders
        m_p, m_sp, m_n, m_p22, m_sp22 = self._mot_encoders_forward(bp_inputs)
        b1, b2, b_n21, b_p12 = self._entire_body_encoder_forward(inputs)
        v1, v2, v_n21, v_p12 = self._view_encoders_forward(inputs)

        # combine encoded parts
        motion_ebds = [m_p, m_sp, m_n]
        body_ebds = [b1, b2]
        view_ebds = [v1, v2]

        # get decoded outputss
        outputs = []  # [p11, p12, p21, p22, sp11, sp12, sp21, sp22, n11, n12, n21, n22]
        for mot in motion_ebds:
            for bd in body_ebds:
                for v in view_ebds:
                    concat_ebd = self._concat_bpe(mot, bd, v)
                    output = [d(concat_ebd[k]) for k, d in enumerate(self.decoders)]
                    outputs.append(output)

        # transpose func : final shape --> [bps, combinations, # of batches, # of joints, # of frames]
        outputs = list(map(list, zip(*outputs)))
        motion_vecs = [m_p, m_sp, m_n, m_p22, m_sp22]
        body_vecs = [b1.reshape(v1.shape[0], -1), b2.reshape(v2.shape[0], -1),
                     b_n21.reshape(v_n21.shape[0], -1), b_p12.reshape(v_p12.shape[0], -1)]
        view_vecs = [v1.reshape(v1.shape[0], -1), v2.reshape(v2.shape[0], -1),
                     v_n21.reshape(v_n21.shape[0], -1), v_p12.reshape(v_p12.shape[0], -1)]

        return outputs, motion_vecs, body_vecs, view_vecs

    def _mot_encoders_forward(self, bp_inputs):
        m_p, m_sp, m_n, m_p22, m_sp22 = [], [], [], [], []
        for k, me in enumerate(self.mot_encoders):
            m_p.append(me(bp_inputs[k][:, 0, 0, 0, :, :]))
            m_sp.append(me(bp_inputs[k][:, 1, 0, 0, :, :]))
            m_n.append(me(bp_inputs[k][:, 2, 1, 1, :, :]))
            m_p22.append(me(bp_inputs[k][:, 0, 1, 1, :, :]))
            m_sp22.append(me(bp_inputs[k][:, 1, 1, 1, :, :]))
        return m_p, m_sp, m_n, m_p22, m_sp22

    def _body_encoders_forward(self, bp_inputs):
        b1, b2, b_n12, b_sp12, b_sp22, b_p21, b_sp21, b_sp11 = [], [], [], [], [], [], [], []
        for k, be in enumerate(self.body_encoders):
            if k == 4:
                b1.append(be(bp_inputs[k][:, 0, 0, 0, :-2, :]))
                b2.append(be(bp_inputs[k][:, 2, 1, 1, :-2, :]))
                b_n12.append(be(bp_inputs[k][:, 2, 0, 1, :-2, :]))
                b_sp12.append(be(bp_inputs[k][:, 1, 0, 1, :-2, :]))
                b_sp22.append(be(bp_inputs[k][:, 1, 1, 1, :-2, :]))
                b_p21.append(be(bp_inputs[k][:, 0, 1, 0, :-2, :]))
                b_sp21.append(be(bp_inputs[k][:, 1, 1, 0, :-2, :]))
                b_sp11.append(be(bp_inputs[k][:, 1, 0, 0, :-2, :]))
            else:
                b1.append(be(bp_inputs[k][:, 0, 0, 0, :, :]))
                b2.append(be(bp_inputs[k][:, 2, 1, 1, :, :]))
                b_n12.append(be(bp_inputs[k][:, 2, 0, 1, :, :]))
                b_sp12.append(be(bp_inputs[k][:, 1, 0, 1, :, :]))
                b_sp22.append(be(bp_inputs[k][:, 1, 1, 1, :, :]))
                b_p21.append(be(bp_inputs[k][:, 0, 1, 0, :, :]))
                b_sp21.append(be(bp_inputs[k][:, 1, 1, 0, :, :]))
                b_sp11.append(be(bp_inputs[k][:, 1, 0, 0, :, :]))
        return b1, b2, b_n12, b_sp12, b_sp22, b_p21, b_sp21, b_sp11
    
    def _entire_body_encoder_forward(self, inputs):
        invis = '_invis' if self.invisibility_augmentation else ''
        b1 = self.body_encoder(inputs['p_1_1' + invis][:, :-2, :])
        b2 = self.body_encoder(inputs['n_2_2' + invis][:, :-2, :])
        b_n21 = self.body_encoder(inputs['n_2_1' + invis][:, :-2, :])
        b_p12 = self.body_encoder(inputs['p_1_2' + invis][:, :-2, :])

        return b1, b2, b_n21, b_p12

    def _view_encoders_forward(self, inputs):
        invis = '_invis' if self.invisibility_augmentation else ''
        v1 = self.view_encoder(inputs['p_1_1' + invis][:, :-2, :])
        v2 = self.view_encoder(inputs['n_2_2' + invis][:, :-2, :])
        v_n21 = self.view_encoder(inputs['n_2_1' + invis][:, :-2, :])
        v_p12 = self.view_encoder(inputs['p_1_2' + invis][:, :-2, :])

        return v1, v2, v_n21, v_p12

    def _concat_bpe(self, m, b, v):
        if self.use_all_joints_on_each_bp:
            concat_mbv = [torch.cat([m[l], b.repeat(1, 1, m[l].shape[-1]), v.repeat(1, 1, m[l].shape[-1])], dim=1)
                          for l in range(len(m))]
        else:
            concat_mbv = [torch.cat([m[l], b[l].repeat(1, 1, m[l].shape[-1]), v.repeat(1, 1, m[l].shape[-1])], dim=1)
                          for l in range(len(m))]
        return concat_mbv

    def _reshape_encoded_inputs(self, input_list):
        reshaped_vecs = []
        for encoded_vec in input_list:
            cur_vecs = [encoded_vec[idx].reshape(encoded_vec[idx].shape[0], -1) for idx in range(len(encoded_vec))]
            reshaped_vecs.append(cur_vecs)
        return reshaped_vecs

    def forward(self, inputs):
        if self.use_all_joints_on_each_bp:
            outputs, motion_vecs, body_vecs, view_vecs = self.cross_with_all_joints_on_each_bp_quadruplet(inputs)
        else:
            outputs, motion_vecs, body_vecs, view_vecs = self.cross_with_quadruplet(inputs)
        return outputs, motion_vecs, body_vecs, view_vecs
