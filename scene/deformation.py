import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from scene.networks import AudioNet, AudioAttNet, MLP
from scene.canonical_tri_plane import canonical_tri_plane
from scene.transformer.transformer import Spatial_Audio_Attention_Module, CrossModalAttentionModule, MultiModalFusion, \
    AttentionModule
from torch.utils.data import DataLoader


# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.grid_pe = grid_pe
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)

        # audio network
        self.audio_in_dim = 29
        self.audio_dim = 32
        self.eye_dim = 1

        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.tri_plane = canonical_tri_plane(args, self.D, self.W)

        self.args = args
        self.only_infer = args.only_infer
        self.enc_x = None
        self.aud_ch_att = None

        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))

        # self.land_net = LdmNet()

        self.MMF = MultiModalFusion(feature_dim=args.d_model)

        self.transformer_one = Spatial_Audio_Attention_Module(args)
        self.transformer_two = CrossModalAttentionModule(args)
        self.transformer_three = AttentionModule(args)

        self.ratio = 0
        self.create_net()

        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)
        self.audio_mlp = MLP(32, args.d_model, 64, 2)

        self.in_dim = 32
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 64, 2)
        self.eye_att_net = MLP(self.in_dim, 1, 16, 2)

        self.eye_mlp = MLP(32, args.d_model, 64, 2)
        self.cam_mlp = MLP(12, args.d_model, 64, 2)
        self.jaw_mlp = MLP(11, args.d_model, 64, 2)
        # self.timenet = MLP(1, args.d_model, 64, 2)
        self.null_vector = nn.Parameter(torch.randn(1, 1, args.d_model))
        self.enc_x_mlp = MLP(32, args.d_model, 64, 2)
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Sobel 卷积核用于边缘检测
        self.sobel_kernel = torch.tensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.timenet = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.time_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb", xyz_max, xyz_min)
        self.tri_plane.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe != 0:

            grid_out_dim = self.grid.feat_dim + (self.grid.feat_dim) * 2
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim, self.W)]

        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.args.d_model, self.W), nn.ReLU(),
                                        nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.args.d_model, self.W), nn.ReLU(),
                                           nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.args.d_model, self.W), nn.ReLU(),
                                              nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.args.d_model, self.W), nn.ReLU(),
                                            nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.args.d_model, self.W), nn.ReLU(),
                                        nn.Linear(self.W, 16 * 3))

    def mlp_init_zeros(self):
        nn.init.xavier_uniform_(self.pos_deform[-1].weight, gain=0.1)
        nn.init.zeros_(self.pos_deform[-1].bias)

        nn.init.xavier_uniform_(self.scales_deform[-1].weight, gain=0.1)
        nn.init.zeros_(self.scales_deform[-1].bias)

        nn.init.xavier_uniform_(self.rotations_deform[-1].weight, gain=0.1)
        nn.init.zeros_(self.rotations_deform[-1].bias)

        nn.init.xavier_uniform_(self.opacity_deform[-1].weight, gain=0.1)
        nn.init.zeros_(self.opacity_deform[-1].bias)

        nn.init.xavier_uniform_(self.shs_deform[-1].weight, gain=0.1)
        nn.init.zeros_(self.shs_deform[-1].bias)

        # triplane
        self.tri_plane.mlp_init_zeros()

    def mlp2cpu(self):
        self.tri_plane.mlp2cpu()

    def eye_encoding(self, value, d_model=32):
        batch_size, _ = value.shape
        value = value.to('cuda')

        positions = torch.arange(d_model, device='cuda').float()
        div_term = torch.pow(10000, (2 * (positions // 2)) / d_model)

        encoded_vec = torch.zeros(batch_size, d_model, device='cuda')
        encoded_vec[:, 0::2] = torch.sin(value * div_term[::2])
        encoded_vec[:, 1::2] = torch.cos(value * div_term[1::2])

        return encoded_vec

    def edge_detection(self, features):
        """
        通用边缘检测模块：
        - 若是图像型特征（4D）：使用 Sobel 边缘检测
        - 若是1D/2D：保留原特征或返回近似的 gradient 增强
        """
        if features.dim() == 4:
            # 图像格式输入: (B, C, H, W)
            B, C, H, W = features.shape
            # 卷积核复制为每个通道一个
            sobel_kernel = self.sobel_kernel.to(features.device).repeat(C, 1, 1, 1)
            return F.conv2d(features, sobel_kernel, padding=1, groups=C)

        elif features.dim() == 2:
            # 向量输入: (B, D)，如 (1, 1) 或 (1, 11)
            B, D = features.shape
            if D < 2:
                return features  # 没法做差分
            grad = torch.abs(features[:, 1:] - features[:, :-1])
            grad = F.pad(grad, (0, 1), mode='replicate')
            return features + grad

        elif features.dim() == 1:
            D = features.shape[0]
            if D < 2:
                return features
            features = features.unsqueeze(0)  # (1, D)
            grad = torch.abs(features[:, 1:] - features[:, :-1])
            grad = F.pad(grad, (0, 1), mode='replicate')
            return (features + grad).squeeze(0)

        else:
            # 其他维度，不支持，直接返回
            return features

    def adaptive_filter(self, audio_features, eye_features, mouth_features):
        """
        自适应滤波模块，根据能量 + mouth 的动态因子，调整滤波强度，并进行边缘加权。
        :param audio_features: 音频特征
        :param eye_features: 眼部特征
        :param mouth_features: 嘴部特征
        :return: 加权后的音频、眼部、嘴部特征
        """
        # 每个特征的能量（L2 范数）
        audio_energy = torch.norm(audio_features, p=2, dim=-1)
        eye_energy = torch.norm(eye_features, p=2, dim=-1)
        mouth_energy = torch.norm(mouth_features, p=2, dim=-1)

        # 动态因子：根据 mouth 特征计算
        dynamic_factor = self.dynamic_mlp(mouth_features)  # (B, 1)

        # 计算滤波权重，alpha 是可学习的，全局乘一个 dynamic 因子（帧级别）
        audio_weight = torch.sigmoid(audio_energy) * self.alpha * dynamic_factor.squeeze(-1)
        eye_weight = torch.sigmoid(eye_energy) * self.alpha * dynamic_factor.squeeze(-1)
        mouth_weight = torch.sigmoid(mouth_energy) * self.alpha * dynamic_factor.squeeze(-1)

        # 应用加权（注意扩展维度匹配）
        weighted_audio = audio_features * audio_weight.unsqueeze(-1)
        weighted_eye = eye_features * eye_weight.unsqueeze(-1)
        weighted_mouth = mouth_features * mouth_weight.unsqueeze(-1)

        # 获取边缘特征
        edge_audio = self.edge_detection(weighted_audio)
        edge_eye = self.edge_detection(weighted_eye)
        edge_mouth = self.edge_detection(weighted_mouth)

        # 将边缘特征与加权特征结合，增强细节
        weighted_audio = weighted_audio + edge_audio
        weighted_eye = weighted_eye + edge_eye
        weighted_mouth = weighted_mouth + edge_mouth

        return weighted_audio, weighted_eye, weighted_mouth

    def query_audio(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features):
        # audio_features [1, 8, 29, 16]) 
        audio_features = audio_features.squeeze(0)
        enc_a = self.audio_net(audio_features)
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))

        if self.only_infer:
            if self.enc_x == None:
                self.enc_x = self.grid(rays_pts_emb[:, :3])
                self.aud_ch_att = self.aud_ch_att_net(self.enc_x)
            enc_x = self.enc_x
            aud_ch_att = self.aud_ch_att

        else:
            enc_x = self.grid(rays_pts_emb[:, :3])  # shape ; # , 32
            aud_ch_att = self.aud_ch_att_net(enc_x)

        enc_a = enc_a.repeat(enc_x.shape[0], 1)

        enc_w = enc_a * aud_ch_att
        eye_att = self.eye_att_net(enc_x)
        eye_att = torch.sigmoid(eye_att)
        eye_features = eye_features * eye_att
        hidden = torch.cat([enc_x, enc_w, eye_features], dim=-1)  # # , 65(32+32+1)

        hidden = self.feature_out(hidden)

        return hidden

    def attention_query_audio_batch(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features,
                                    cam_features, jaw_features, time_features, time_emb):
        # audio_features [B, 8, 29, 16])

        B, _, _, _ = audio_features.shape

        if self.only_infer:  # cashing
            if self.enc_x == None:
                self.enc_x = self.tri_plane(rays_pts_emb, time_emb, only_feature=True, train_tri_plane=self.args.train_tri_plane)
            enc_x = self.enc_x[:B]

        else:
            # audio_features [B, 8, 29, 16]) 
            enc_x = self.tri_plane(rays_pts_emb, time_emb, only_feature=True, train_tri_plane=self.args.train_tri_plane)

        enc_a_list = []
        for i in range(B):
            enc_a = self.audio_net(audio_features[i])  # 8 32
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # 1 32
            enc_a_list.append(enc_a.unsqueeze(0))

        enc_a = torch.cat(enc_a_list, dim=0)  # B, 1, 32
        enc_eye = self.eye_encoding(eye_features).unsqueeze(1)  # B, 1, 32

        # enc_landmark = self.land_net(self.args.landmarks).unsqueeze(1)
        enc_cam = self.cam_mlp(cam_features).unsqueeze(1)

        if isinstance(jaw_features, torch.Tensor):
            jaw_features = [jaw_features]
        jaw_features = torch.stack(jaw_features)  # 将它们堆叠成一个形状为 (batch_size, 9) 的张量

        # 现在 jaw_features 是一个 PyTorch 张量，形状为 (batch_size, 9)
        jaw_features = jaw_features.float()  # 确保数据类型是 float32

        # 传入 jaw_mlp 进行计算
        enc_jaw = self.jaw_mlp(jaw_features).unsqueeze(1)  # 输出形状应为 (batch_size, 1, 64)

        if self.args.d_model != 32:
            enc_a = self.audio_mlp(enc_a)  # B, 1, dim
            enc_eye = self.eye_mlp(enc_eye)  # B, 1, dim

        enc_time = torch.stack(time_emb)  # shape: [1] → torch.Size([1])
        enc_time = self.timenet(enc_time.unsqueeze(1))  # shape: [1, 1] → [1, 64]
        enc_time = enc_time.unsqueeze(1)  # [B, 1, d_model]
        enc_source = torch.cat([
            enc_a,  # [B, 1, d_model]
            enc_eye,  # [B, 1, d_model]
            enc_cam,  # [B, 1, d_model]
            enc_jaw,  # [B, 1, d_model]
            self.null_vector.repeat(B, 1, 1)
        ], dim=1)  # [B, 5 d_model]
        # 多模态特征拼接，添加一层融合模块
        enc_source = self.MMF(enc_source)
        # 交叉注意力机制

        # x, attention = self.transformer_two(enc_x, enc_source)
        # 普通注意力
        x, attention = self.transformer_three(enc_x, enc_source, enc_time)
        # x, attention = self.transformer_one(enc_x, enc_source, enc_time)
        return x, attention, enc_a

    def attention_query_audio(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features,
                              jaw_features, time_features, time_emb):
        # audio_features [1, 8, 29, 16])
        if self.only_infer:
            if self.enc_x == None:
                enc_x = self.tri_plane(rays_pts_emb.unsqueeze(0), only_feature=True,
                                       train_tri_plane=self.args.train_tri_plane).unsqueeze(0)
            enc_x = self.enc_x

        else:
            enc_x = self.tri_plane(rays_pts_emb.unsqueeze(0), only_feature=True,
                                   train_tri_plane=self.args.train_tri_plane).unsqueeze(0)

        # 超参数w 训练
        # enc_x = self.Mouth_mask_w(enc_x, rays_pts_emb, mouth_mask)

        audio_features = audio_features.squeeze(0)  # 8, 29, 16
        enc_a = self.audio_net(audio_features)  # 8, 32
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)).unsqueeze(0)  # 1, 1, 32

        enc_eye = self.eye_encoding(eye_features).unsqueeze(1)  # 1, 1, 32

        if isinstance(jaw_features, torch.Tensor):
            jaw_features = [jaw_features]
        jaw_features = torch.stack(jaw_features)  # 将它们堆叠成一个形状为 (batch_size, 9) 的张量

        # 现在 jaw_features 是一个 PyTorch 张量，形状为 (batch_size, 9)
        jaw_features = jaw_features.float()  # 确保数据类型是 float32

        # 传入 jaw_mlp 进行计算
        enc_jaw = self.jaw_mlp(jaw_features).unsqueeze(1)  # 输出形状应为 (batch_size, 1, 64)

        if self.args.d_model != 32:
            enc_a = self.audio_mlp(enc_a)  # B, 1, dim
            enc_eye = self.eye_mlp(enc_eye)  # B, 1, dim

        enc_time = torch.stack(time_emb)  # shape: [1] → torch.Size([1])
        enc_time = self.timenet(enc_time.unsqueeze(1))  # shape: [1, 1] → [1, 64]
        enc_time = enc_time.unsqueeze(1)  # [B, 1, d_model]
        enc_source = torch.cat([
            enc_a,  # [B, 1, d_model]
            enc_eye,  # [B, 1, d_model]
            enc_jaw,  # [B, 1, d_model]
        ], dim=1)  # [B, 5 d_model]

        x, attention = self.transformer_one(enc_x, enc_source, enc_time)  # 1, N, dim
        # x, attention = self.transformer_two(enc_x, enc_source)  # 1, N, dim
        # x, attention = self.transformer_three(enc_x, enc_source)
        # attention = attention * mouth_mask
        x = x.squeeze()

        return x, attention

    @property
    def get_empty_ratio(self):
        return self.ratio

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity=None, shs_emb=None,
                audio_features=None, eye_features=None, cam_features=None, mouth_feature=None, time_features=None,
                time_emb=None):
        if audio_features is None:
            return self.forward_static(rays_pts_emb, time_emb)
        elif len(rays_pts_emb.shape) == 3:
            return self.forward_dynamic_batch(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, audio_features,
                                              eye_features, cam_features, mouth_feature, time_features, time_emb)
        elif len(rays_pts_emb.shape) == 2:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, audio_features,
                                        eye_features, mouth_feature, time_features, time_emb)

    def forward_static(self, rays_pts_emb, time_emb):
        grid_feature, scale, rotation, opacity, sh = self.tri_plane(rays_pts_emb, time_emb)
        # dx = self.static_mlp(grid_feature)
        return rays_pts_emb, scale, rotation, opacity, sh.reshape([sh.shape[0], sh.shape[1], 16, 3])

    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, audio_features,
                        eye_features, jaw_features, time_features, time_emb):
        # 调用自适应滤波
        # weighted_audio, weighted_eye, weighted_mouth = self.adaptive_filter(
        #     audio_features, eye_features, jaw_features
        # )

        hidden, attention = self.attention_query_audio(rays_pts_emb, scales_emb, rotations_emb, audio_features,
                                                       eye_features, jaw_features, time_features, time_emb)

        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:, :3])
        else:
            mask = torch.ones_like(opacity_emb[:, 0]).unsqueeze(-1)

        if self.args.no_dx:
            pts = rays_pts_emb[:, :3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:, :3])
            pts = rays_pts_emb[:, :3] * mask + dx
        if self.args.no_ds:

            scales = scales_emb[:, :3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:, :3])
            scales = scales_emb[:, :3] * mask + ds

        if self.args.no_dr:
            rotations = rotations_emb[:, :4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:, :4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:, :4] + dr

        if self.args.no_do:
            opacity = opacity_emb[:, :1]
        else:
            do = self.opacity_deform(hidden)

            opacity = torch.zeros_like(opacity_emb[:, :1])
            opacity = opacity_emb[:, :1] * mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3])

            shs = torch.zeros_like(shs_emb)
            shs = shs_emb * mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs, attention

    def forward_dynamic_batch(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, audio_features,
                              eye_features, cam_features, jaw_features, time_features, time_emb):
        # 调用自适应滤波
        # weighted_audio, weighted_eye, weighted_mouth = self.adaptive_filter(
        #     audio_features, eye_features, jaw_features
        # )
        hidden, attention, enc_a = self.attention_query_audio_batch(rays_pts_emb, scales_emb, rotations_emb,
                                                                    audio_features, eye_features, cam_features,
                                                                    jaw_features, time_features, time_emb)
        time_tensor = torch.stack(time_emb, dim=0).to(rays_pts_emb.device)  # (B,)
        enc_time = self.timenet(time_tensor.unsqueeze(1))  # (B, 1) -> (B, 64)

        B, _, _, _ = audio_features.shape
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:, :3])
        else:
            mask = torch.ones_like(opacity_emb[:, :, 0]).unsqueeze(-1)
        if self.args.no_dx:
            pts = rays_pts_emb[:, :, :3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:, :, :3])
            pts = rays_pts_emb[:B, :, :3] * mask + dx
            # pts = rays_pts_emb[:B, :, :3] * mask + dx + pts_m
        if self.args.no_ds:

            scales = scales_emb[:, :, :3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:, :, :3])
            scales = scales_emb[:B, :, :3] * mask + ds
            # scales = scales_emb[:B, :, :3] * mask + ds + scales_m

        if self.args.no_dr:
            rotations = rotations_emb[:, :, :4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:, :, :4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:B, :, :4] + dr
                # rotations = rotations_emb[:B, :, :4] + dr + rotations_m

        if self.args.no_do:
            opacity = opacity_emb[:, :, :1]
        else:
            do = self.opacity_deform(hidden)

            opacity = torch.zeros_like(opacity_emb[:, :, :1])
            opacity = opacity_emb[:B, :, :1] * mask + do
            # opacity = opacity_emb[:B, :, :1] * mask + do + opacity_m
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0], shs_emb.shape[1], 16, 3])

            shs = torch.zeros_like(shs_emb)
            shs = shs_emb[:B] * mask.unsqueeze(-1) + dshs
            # shs = shs_emb[:B] * mask.unsqueeze(-1) + dshs + shs_m.reshape([shs_emb.shape[0], shs_emb.shape[1], 16, 3])

        return pts, scales, rotations, opacity, shs, attention

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                parameter_list.append(param)
        return parameter_list

    def train_diffusion_model(model, diffusion, dataloader, optimizer, epochs=10, device='cuda'):
        model.to(device)
        diffusion_helper = diffusion

        for epoch in range(epochs):
            for batch in dataloader:
                # 获取输入数据
                image = batch['image'].to(device)  # 输入图像
                mask = batch['mouth_mask'].to(device)  # 嘴部 mask
                audio = batch['audio'].to(device)  # 音频特征
                x = batch['x'].to(device)  # 嘴部位置编码
                mu_real = batch['mu'].to(device)  # 真实嘴部偏移

                # 随机选择时间步
                t = torch.randint(0, diffusion_helper.timesteps, (image.size(0),), device=device)

                # 添加噪声到目标
                noisy_mu, noise = diffusion_helper.add_noise(mu_real, t)

                # 预测噪声
                mu_pred = model(image, mask, audio, x)
                loss = F.mse_loss(mu_pred, noise)

                # 优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


class deform_network(nn.Module):
    def __init__(self, args):
        super(deform_network, self).__init__()
        net_width = args.net_width
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        grid_pe = args.grid_pe

        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3) + (3 * (posbase_pe)) * 2,
                                           grid_pe=grid_pe, args=args)
        self.only_infer = args.only_infer
        self.register_buffer('pos_poc', torch.FloatTensor([(2 ** i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2 ** i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2 ** i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        self.deformation_net.mlp_init_zeros()

        self.point_emb = None
        self.scales_emb = None
        self.rotations_emb = None
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, audio_features=None,
                eye_features=None, cam_features=None, jaw_features=None, times_sel=None):
        if audio_features is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, shs, audio_features, eye_features,
                                        cam_features, jaw_features, times_sel)
        else:
            return self.forward_static(point, scales, rotations, opacity, shs, times_sel)

    @property
    def get_aabb(self):

        return self.deformation_net.get_aabb

    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio

    def forward_static(self, points, scales, rotations, opacity, shs, times_sel):
        points = self.deformation_net(points, time_emb=times_sel)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, audio_features=None,
                        eye_features=None, cam_features=None, jaw_features=None, times_sel=None, is_silent=True):
        if self.only_infer:
            if self.point_emb == None:
                self.point_emb = poc_fre(point, self.pos_poc)  # #, 3 -> #, 63
                self.scales_emb = poc_fre(scales, self.rotation_scaling_poc)
                self.rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)

            point_emb = self.point_emb
            scales_emb = self.scales_emb
            rotations_emb = self.rotations_emb

            means3D, scales, rotations, opacity, shs, attention = self.deformation_net(point_emb,
                                                                                       scales_emb,
                                                                                       rotations_emb,
                                                                                       opacity,
                                                                                       shs,
                                                                                       audio_features, eye_features,
                                                                                       cam_features, jaw_features, None,
                                                                                       times_sel)
            return means3D, scales, rotations, opacity, shs, attention

        else:
            point_emb = poc_fre(point, self.pos_poc)  # B, N, 3 -> B, N, 63
            scales_emb = poc_fre(scales, self.rotation_scaling_poc)
            rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
            means3D, scales, rotations, opacity, shs, attention = self.deformation_net(point_emb,
                                                                                       scales_emb,
                                                                                       rotations_emb,
                                                                                       opacity,
                                                                                       shs,
                                                                                       audio_features, eye_features,
                                                                                       cam_features, jaw_features, None,
                                                                                       times_sel)
            return means3D, scales, rotations, opacity, shs, attention

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

    def mlp2cpu(self):
        self.deformation_net.mlp2cpu()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight, gain=1)


def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb
