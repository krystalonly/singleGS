import torch
import torch.nn as nn
from scene.hexplane import HexPlaneField


class canonical_tri_plane(nn.Module):
    def __init__(self, args=None, D=8, W=64):
        super(canonical_tri_plane, self).__init__()
        self.W = W
        self.D = D
        self.args = args
        self.no_grid = args.no_grid

        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        input_dim = self.grid.feat_dim + args.d_model  # 原有 grid feat + time_emb

        self.time_grid_proj = nn.Linear(input_dim, 64)  # 新加的降维层

        self.feature_out = [nn.Linear(64, self.W)]  # 注意从64而不是input_dim开始
        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        # self.feature_out = [nn.Linear(input_dim, self.W)]
        self.offset_mlp = nn.Sequential(
            nn.Linear(3, W),
            nn.ReLU(),
            nn.Linear(W, 3)
        )
        self.timenet = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.grid_feat_reduce = nn.Linear(64, 3)

        # for i in range(self.D - 1):
        #     self.feature_out.append(nn.ReLU())
        #     self.feature_out.append(nn.Linear(self.W, self.W))
        # self.feature_out = nn.Sequential(*self.feature_out)

        self.scales = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 2))
        self.rotations = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4))
        self.opacity = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))
        self.shs = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 16 * 3))

    def mlp_init_zeros(self):
        for layer in [self.scales, self.rotations, self.opacity, self.shs]:
            nn.init.xavier_uniform_(layer[-1].weight, gain=0.1)
            nn.init.zeros_(layer[-1].bias)

    def mlp2cpu(self):
        self.feature_out = self.feature_out.to('cpu')
        self.scales = self.scales.to('cpu')
        self.rotations = self.rotations.to('cpu')
        self.opacity = self.opacity.to('cpu')
        self.shs = self.shs.to('cpu')

    def forward(self, rays_pts_emb, time_emb=None, only_feature=False, train_tri_plane=True):
        """
        time_emb: Tensor[B, D]，必须是 timenet 输出维度，与 d_model 对齐
        """
        B, N, _ = rays_pts_emb.shape
        coords = rays_pts_emb[0, :, :3]  # (N, 3)

        # 1. 时间特征编码
        if time_emb is not None:
            time_tensor = torch.stack(time_emb).to(coords.device)  # [B]
            time_feat = self.timenet(time_tensor.unsqueeze(1))  # [B, 64]
            time_feat_expand = time_feat.unsqueeze(1).expand(-1, coords.shape[0], -1)  # [B, N, 64]
        else:
            time_feat_expand = torch.zeros(B, coords.shape[0], 64, device=coords.device)  # fallback

        # 2. 原始坐标 → 三平面grid特征
        grid_feat = self.grid(coords).unsqueeze(0).repeat(B, 1, 1)  # [B, N, grid_feat_dim]

        # 3. 拼接时间 → 降维融合
        # grid_feat = torch.cat([grid_feat, time_feat_expand], dim=-1)  # [B, N, grid+64]
        # grid_feat = self.time_grid_proj(grid_feat)  # [B, N, 64]

        # 4. 用融合后的特征预测坐标扰动 offset（每帧不同）
        # offset_mlp 输入为原始坐标，也可以改为输入融合后特征
        offset = self.offset_mlp(coords)  # 或 offset = self.offset_mlp(grid_feat[0])
        adjusted_coords = coords + offset

        # 5. 用扰动后的坐标再查一次 grid 特征（如果你希望 offset 真的参与影响）
        # grid_feat_reduced = self.grid_feat_reduce(grid_feat)
        final_grid_feat = self.grid(adjusted_coords).unsqueeze(0).repeat(B, 1, 1)  # [B, N, grid_feat_dim]

        # 6. 拼接时间，再次融合
        final_grid_feat = torch.cat([final_grid_feat, time_feat_expand], dim=-1)
        final_grid_feat = self.time_grid_proj(final_grid_feat)

        # 7. 后续属性预测
        if only_feature:
            return final_grid_feat.detach() if not train_tri_plane else final_grid_feat

        feature = self.feature_out(final_grid_feat)
        scale = self.scales(feature)
        rotation = self.rotations(feature)
        opacity = self.opacity(feature)
        sh = self.shs(feature)

        return feature, scale, rotation, opacity, sh
