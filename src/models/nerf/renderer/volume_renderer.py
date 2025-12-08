import numpy as np
import torch
import torch.nn.functional as F
from src.config import cfg

class Renderer:
    def __init__(self, net):
        """
        Write your codes here.
        """
        self.net=net
        self.N_samples = net.N_samples #粗模型每条光线上的采样点数
        self.N_importance = net.N_importance  #细模型每条光线上的采样点数
        self.chunk = net.chunk
        self.batch_size = net.batch_size  # 也就是cfg.task_arg.N_rays，是1024
        self.white_bkgd = net.white_bkgd
        self.use_viewdirs = net.use_viewdirs
        self.device = net.device

        self.model=net.model.to(self.device)  #必须显式地指定设备，否则模型的所有参数默认是存在CPU中的
        self.model_fine=net.model_fine.to(self.device)

        self.t_near=getattr(cfg.task_arg,"t_near",0.0)
        self.t_far=getattr(cfg.task_arg,"t_far",6.0)  # ❓我在cfg里没看到这两个设定。AI说一般取0.0和6.0




    # 在光线的近平面t_near和远平面t_far之间生成N_samples个采样点。
    # r(t)=o+t*d
    def _sample_rays(self,rays_o,rays_d,N_samples,
                     perturb=True,lindisp=False):   #默认进行分层随机采样、深度均匀采样
        # rays_o,rays_d都是(N_rays,3)
        N_rays=rays_o.shape[0]

        tau=torch.linspace(0.0 , 1.0, N_samples,device=self.device)
        if lindisp==True:
            tmp=tau*(1/self.t_far)+(1-tau)*(1/self.t_near)
            t=1/tmp
        else:
            t=tau*self.t_far+(1-tau)*self.t_near   
            # 共(N_samples,)个点，从t_far均匀到t_near均匀分布，“栅栏区间”
        #如果perturb==False，那么这已经是采样结果了
        #如果perturb==True，需要在这样的栅栏区间内随机取点
        
        if perturb:
            mids=(t[1:]+t[:-1])/2   #(N_samples-1,)   存储了所有栅栏区间的中点
            upper=torch.cat([mids,t[-1:]],dim=-1)  # (N_samples,)
            lower=torch.cat([t[:1],mids],dim=-1) # (N_samples,)

            #在每个区间 [lower, upper] 内随机均匀采样
            tau2=torch.rand(N_samples,device=self.device)  # 0.0到1.0之间
            t=lower+tau2*(upper-lower)  # (N_samples,)

        # 扩展到每一条光线
        t=t.expand([N_rays,N_samples])  # (N_rays,N_samples)
        t_1=t.unsqueeze(dim=-1)  # (N_rays,N_samples,1)

        pts=rays_o[:,None]+t_1*rays_d[:,None]
        # rays_o[:,None]是(N_rays, 1, 3)
        # pts是(N_rays,N_samples,3)
        
        return pts,t
        # (N_rays,N_samples,3)    (N_rays,N_samples)





    #将NeRF预测的原始值σ，c转化为最终的像素颜色C(r)和深度D(r)。
    def _raw2output(self,raw,t,rays_d,white_bkgd=False):
        '''
        raw是Network的原始输出 (N_rays, N_samples, 4)
        t是分层随机采样使用的深度  (N_rays,N_samples)
        rays_d 是 (N_rays,3)
        '''
        N_rays,N_samples,_=raw.shape

        deltas=t[:,1:]-t[:,:-1]
        deltas=torch.cat([deltas,1e10*torch.ones_like(t[:,:1])],dim=-1)  #(N_rays,N_samples)
        # 最远的那个区间，认为区间距离是无限大
        rays_d_norm=torch.norm(rays_d[:,None,:],dim=-1)  # (N_rays,1)
        # 乘上d的模长才是真正的空间距离
        deltas=deltas*rays_d_norm # (N_rays,N_samples)


        c=torch.sigmoid(raw[:,:,:3])  # (N_rays, N_samples, 4) 保证值在[0,1]之间
        sigma=F.relu(raw[:,:,3])  #(N_rays, N_samples) 保证体密度非负

        tmp=torch.exp(-sigma*deltas)  #(N_rays, N_samples)
        tmp_start=torch.ones([N_rays,1],device=self.device)
        tmp=torch.cat([tmp_start,tmp[:,:-1]],dim=-1)   # (N_rays, N_samples)
        T=torch.cumprod(tmp,dim=-1)  # 使用累乘得到光线在这个点的透过率

        weights=T*(1-torch.exp(-sigma*deltas))  # (N_rays, N_samples)
        C=(weights[:,:,None]*c).sum(dim=1)  # 最终的像素颜色C(r)    (N_rays,3)
        D=(weights*t).sum(dim=-1)  # depthmap D(r)    (N_rays)

        if white_bkgd:
            tmp=weights.sum(dim=-1,keepdim=True)  # (N_rays,1) 是一条光线的累积不透明度
            C=C+1.0*(1-tmp)   # 如果tmp=0，说明应该是背景的颜色（白色）

        return C,D,weights



    def render(self, batch):
        """
        Write your codes here.
        """
        pass
