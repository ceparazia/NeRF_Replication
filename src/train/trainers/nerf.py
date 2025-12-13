import torch
import torch.nn as nn
from src.models.nerf.renderer.volume_renderer import Renderer


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = Renderer(self.net)

        # add metrics here
        self.loss_fn=nn.MSELoss(reduction="mean")

    def forward(self, batch):
        """
        Write your codes here.
        """
        rays_o=batch['rays_o']  # batch是通过Dataset.getitem得到的字典
        rays_d=batch['rays_d']
        rgbs_gt=batch['rgbs']
        # (N_rays,3)  (N_rays,3)  (N_rays,3)

        render_ret=self.renderer.render(batch)
        
        C_coarse=render_ret["C_coarse"]
        D_coarse=render_ret["D_coarse"]
        C_fine=render_ret["C_fine"]
        D_fine=render_ret["D_fine"]
        weights_fine=render_ret["weights_fine"]

        loss_coarse=self.loss_fn(C_coarse,rgbs_gt)
        loss_fine=self.loss_fn(C_fine,rgbs_gt)
        loss=loss_coarse+loss_fine

        output={
            'C_coarse':C_coarse,
            'C_fine':C_fine,
            'rgbs_gt':rgbs_gt,
            'D_coarse':D_coarse,
            'D_fine':D_fine,
            'weights_fine':weights_fine
        }


        # 这里要匹配/home/qiuminghao/nerf-replication/src/train/trainers/trainer.py里面的接口
        # NetworkWrapper类的forward方法应当有4个返回值
        loss_stats={
            "loss":loss.detach(),  # 后续接口会求loss.mean()。所以先切断梯度追踪
            "loss_coarse":loss_coarse.detach(),
            "loss_fine":loss_fine.detach()
        }

        image_stats={
            # 还没想好
        }


        return output,loss,loss_stats,image_stats

