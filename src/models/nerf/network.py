import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from src.models.encoding import get_encoder
from src.config import cfg


class NeRF(nn.Module):   # 位置x',光线方向d' -> 形体σ,颜色c
    def __init__(
        self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_viewdirs=False
    ): 
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch  #位置编码特征长度
        self.input_ch_views = input_ch_views #视图编码特征长度

        self.skips = skips
        #skips=4表示在位置网络的第 4 层结束时，
        #会将输出(W,)与原始输入(input_ch,)拼接起来，变成(W+input_ch)
        #这样可以确保深层网络能够更直接地利用低层级的空间信息。


        # σ只由x得到
        # 完整的NeRF中，c需要由x,d共同得到
        self.use_viewdirs = use_viewdirs
        # 这决定了网络是否将d作为输入
        # 如果 use_viewdirs=False：模型假设颜色c只由位置x决定，与光线方向d无关。这无法表现高光。
        
        
        
        self.output_ch = 5 if self.use_viewdirs else 4
        # 当 use_viewdirs 为 True 时，网络结构会分叉，
        # 并增加专门处理视角方向的 views_linears。
        # 同时，self.output_ch 变为 5 。因为RGB 3个 + σ 1 个 + 1个长度W的特征向量
        # 否则不含特征向量为 4。
        # ❓但是这个5好像没用





        #位置网络：x -> σ和1个长度为W的特征向量
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)]
            + [
                (
                    nn.Linear(self.W, self.W)
                    if i not in self.skips
                    else nn.Linear(self.W + self.input_ch, self.W)
                )
                for i in range(self.D - 1)
            ]
        )


        #视图网络：d和长度为W的特征向量 -> c
        self.views_linears = nn.ModuleList(
            [nn.Linear(self.input_ch_views + self.W, self.W // 2)]
            # 视图网络通常比位置网络短得多（例如，位置网络 8 层，视图网络 1 层）。
            # 将维度降到W//2,可以减少视图网络中的参数量和计算量
        )

        if self.use_viewdirs:
            # feature vector(256)
            self.feature_linear = nn.Linear(self.W, self.W)
            # alpha(1)  ❓这里应该是σ？
            self.alpha_linear = nn.Linear(self.W, 1)
            # rgb color(3)
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            # output channel(default: 4)
            self.output_linear = nn.Linear(self.W, self.output_ch)

    def forward(self, x):  #输入的x是由位置编码x'和方向编码d'拼接成的，形状(N,input_ch+input_ch_views)
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        # 分割之后，input_pts是(N,input_ch)
        # input_views是(N,input_ch_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):   #位置网络
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # 此时h已经是初始x'通过8层全连接层后得到的一个长度为W的特征向量，形状(N,W)
        if self.use_viewdirs:   #此时输出的outputs
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)  #(N,W),携带着x'的信息
            h = torch.cat([feature, input_views], -1)  #(N,W+input_ch_views),带着x',d'的信息

            for i, l in enumerate(self.views_linears):  #视图网络
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)  #(N,3)
            outputs = torch.cat([rgb, alpha], -1)  #(N,4)
        else:  #不使用views_linear，直接用h得到color
            outputs = self.output_linear(h)   #(N,4)

        return outputs   # 返回了c和α

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears])
            )
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1])
            )

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear])
        )
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1])
        )

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears])
        )
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1])
        )

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear])
        )
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1])
        )

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear])
        )
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1])
        )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.N_samples = cfg.task_arg.N_samples  #粗模型每条光线上的采样点数
        self.N_importance = cfg.task_arg.N_importance  #细模型每条光线上的采样点数
        self.chunk = cfg.task_arg.chunk_size
        self.batch_size = cfg.task_arg.N_rays
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.use_viewdirs = cfg.task_arg.use_viewdirs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder 这些是位置编码的设置
        self.embed_fn, self.input_ch = get_encoder(cfg.network.xyz_encoder)
        self.embeddirs_fn, self.input_ch_views = get_encoder(cfg.network.dir_encoder)

        # coarse model
        self.model = NeRF(
            D=cfg.network.nerf.D,
            W=cfg.network.nerf.W,
            input_ch=self.input_ch,
            input_ch_views=self.input_ch_views,
            skips=cfg.network.nerf.skips,
            use_viewdirs=self.use_viewdirs,
        )

        # fine model
        self.model_fine = NeRF(
            D=cfg.network.nerf.D,
            W=cfg.network.nerf.W,
            input_ch=self.input_ch,
            input_ch_views=self.input_ch_views,
            skips=cfg.network.nerf.skips,
            use_viewdirs=self.use_viewdirs,
        )

    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""

        def ret(inputs):
            return torch.cat(
                [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
            )

        return ret

    def forward(self, inputs, viewdirs, model=""):
        """Prepares inputs and applies network 'fn'."""
        if model == "fine":
            fn = self.model_fine
        else:
            fn = self.model

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        # inputs 是沿着光线采样的采样点的 3D 坐标x，形状为(N_rays,N_samples,3)
        # 为了送入MLP，需要将其展平为(N_rays*N_samples,3)
        embedded = self.embed_fn(inputs_flat)  # 3D向量x -> 位置编码后的高维特征向量x'，长度input_ch
        # (N_rays*N_samples , input_ch)


        if self.use_viewdirs:
            # viewdirs的形状是(N_rays,3)
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            # 将每条光线的方向信息，扩展到该光线上的每个采样点，再展平
            # viewdirs[:, None]是(N_rays,1,3)
            # input_dirs是(N_rays,N_samples,3)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            # 展平成(N_rays*N_samples,3)
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)  # d -> d',长度input_ch_views
            # 然后进行位置编码，为 NeRF 模型输入做准备
            embedded = torch.cat([embedded, embedded_dirs], -1)
            # 拼接x'和d',得到(N_rays*N_samples , input_ch+input_ch_views)


        # embeddded就是要喂给NeRF的完整输入
        embedded = embedded.to(torch.float32)
        outputs_flat = self.batchify(fn, self.chunk)(embedded)
        # 把输入分块，喂给NeRF，再把结果拼起来，得到outputs_flat，形状是(N,4)。此处N=N_rays*N_samples
        outputs = torch.reshape(
            outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )  # 重新塑形为(N_rays,N_samples,4)，就得到了每条光线、每个采样点对应的σ和c
        return outputs
