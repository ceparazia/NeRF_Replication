import numpy as np
import torch
import torch.nn.functional as F
from src.config import cfg
from src.models.nerf.renderer import build_occupancyQ
import os

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

        self.pre_chunk_size=cfg.task_arg.pre_chunk_size
        self.perturb= cfg.task_arg.perturb
        self.lindisp= cfg.task_arg.lindisp
        self.double_chunk_coarse=cfg.task_arg.double_chunk_coarse
        self.double_chunk_fine=cfg.task_arg.double_chunk_fine
        
        self.use_ERT= cfg.task_arg.use_ERT   # ❓ 应该设置成训练模式禁用ERT，测试模式使用ERT
        self.ERT_threshold= cfg.task_arg.ERT_threshold


        self.use_ESS= cfg.task_arg.use_ESS
        if self.use_ESS:
            occ_filename=cfg.task_arg.occ_filename
            save_dir="data/Q_occ_grid for ESS/"
            occ_path=os.path.join(save_dir,f"{occ_filename}.pth")
            try:
                self.occ_grid=torch.load(occ_path).to(self.device)
                print(f"使用ESS。加载已有的occupancy grid:{occ_filename}.pth")
            except FileNotFoundError:
                print(f"使用ESS。但是未找到{occ_filename}.pth")
                print(f"正在生成{occ_filename}.pth")
                build_occupancyQ.build_occupancy(net)

                try:
                    self.occ_grid=torch.load(occ_path).to(self.device)
                    print(f"生成完毕。已加载{occ_filename}.pth")
                except FileNotFoundError:
                    print(f"生成失败。禁用ESS")

            if self.occ_grid==None:
                self.use_ESS=False


        self.model=net.model.to(self.device)  #必须显式地指定设备，否则模型的所有参数默认是存在CPU中的
        self.model_fine=net.model_fine.to(self.device)

        self.t_near=getattr(cfg.task_arg,"t_near",2.0)
        self.t_far=getattr(cfg.task_arg,"t_far",6.0)  # ❓我在cfg里没看到这两个设定。AI说一般取0.0和6.0




    # 在光线的近平面t_near和远平面t_far之间生成N_samples个采样点。
    # r(t)=o+t*d
    def _sample_rays(self,rays_o,rays_d,N_samples,
                     perturb=True,lindisp=False):   #默认进行分层随机采样、深度均匀采样
        # rays_o,rays_d都是(N_rays,3)
        N_rays=rays_o.shape[0]


        if self.use_ESS:
            pass  # 会需要用到self.occ_grid

        else:
            tau=torch.linspace(0.0 , 1.0, N_samples,device=self.device)
            if lindisp==True:
                tmp=tau*(1/self.t_far)+(1-tau)*(1/self.t_near)
                t=1/tmp
            else:
                t=tau*self.t_far+(1-tau)*self.t_near   
                # 共(N_samples,)个点，从t_far均匀到t_near均匀分布，“栅栏区间”
            # 如果perturb==False，那么这已经是采样结果了
            # 如果perturb==True，需要再在这样的栅栏区间内随机取点
            

            # ❓ 或许应该先扩展t，再生成(N_rays,N_samples)形状的随机矩阵tau2?
            # 但是那样会引入过多计算开销
            mids=(t[1:]+t[:-1])/2   #(N_samples-1,)   存储了所有栅栏区间的中点
            if perturb:
                upper=torch.cat([mids,t[-1:]],dim=-1)  # (N_samples,)
                lower=torch.cat([t[:1],mids],dim=-1) # (N_samples,)

                #在每个区间 [lower, upper] 内随机均匀采样
                tau2=torch.rand(N_samples,device=self.device)  # 0.0到1.0之间
                t=lower+tau2*(upper-lower)  # (N_samples,)

            # 采样仅仅由t_near, t_far, N_samples决定，与具体的光线无关
            # 因此可以让所有光线共享同一组深度值t
            # 扩展到每一条光线
            t=t.expand([N_rays,N_samples])  # (N_rays,N_samples)
            t_1=t.unsqueeze(dim=-1)  # (N_rays,N_samples,1)

            # import ipdb; ipdb.set_trace()
            pts=rays_o[:,None]+t_1*rays_d[:,None]
            # 所有光线共享同一组深度值，但是每条光线的rays_d向量不同
            # rays_o[:,None]是(N_rays, 1, 3)
            # pts是采样点的三维坐标，形状(N_rays,N_samples,3)
            
        return pts,t
        # (N_rays,N_samples,3)    (N_rays,N_samples)





    #将NeRF预测的原始值σ，c转化为最终的像素颜色C(r)和深度D(r)。
    def _raw2output(self,raw,t,rays_d,white_bkgd=False):
        '''
        raw是Network的原始输出 (N_rays, N_samples, 4)  在长度为4的维度上的c和σ拼接出来的
        t是分层随机采样使用的深度  (N_rays,N_samples)
        rays_d 是 (N_rays,3)
        '''
        N_rays,N_samples,_=raw.shape

        deltas=t[:,1:]-t[:,:-1]  #深度区间的差
        deltas=torch.cat([deltas,1e10*torch.ones_like(t[:,:1])],dim=-1)  #(N_rays,N_samples)
        # 最远的那个区间，认为区间距离是无限大
        rays_d_norm=torch.norm(rays_d[:,None,:],dim=-1)  # (N_rays,1)
        # 乘上d的模长才是真正的空间距离
        deltas=deltas*rays_d_norm # (N_rays,N_samples)


        c=torch.sigmoid(raw[:,:,:3])  # (N_rays, N_samples, 3) 保证值在[0,1]之间
        sigma=F.relu(raw[:,:,3])  #(N_rays, N_samples) 保证体密度非负

        tmp=torch.exp(-sigma*deltas)  #(N_rays, N_samples)
        tmp_start=torch.ones([N_rays,1],device=self.device)
        tmp=torch.cat([tmp_start,tmp[:,:-1]],dim=-1)   # (N_rays, N_samples)
        T=torch.cumprod(tmp,dim=-1)  # T就是“到达这个点还有多少光”
        # T是从1递减到0的

        weights=T*(1-torch.exp(-sigma*deltas))  # (N_rays, N_samples)
        # weights越大，表示这个采样点将对最终渲染出来的RGB的贡献越大


        if self.use_ERT:
            mask= T<self.ERT_threshold  # (N_rays, N_samples) bool型
            mask= mask.int()
            mask= mask.cumsum(dim=1)  # 虽然T_i应当是递减的，但是可能出现浮点数精度温度。
            # 使用累加，确保若T_i<阈值了，则后面所有点的T都要被terminate
            discard_mask=(mask>0)  # (N_rays, N_samples) bool型
            weights[discard_mask]=0.0

        C=(weights[:,:,None]*c).sum(dim=1)  # 最终的像素颜色C(r)    (N_rays,3)
        D=(weights*t).sum(dim=-1)  # depthmap D(r)    (N_rays)

        if white_bkgd:
            tmp=weights.sum(dim=-1,keepdim=True)  # (N_rays,1) 是每条光线的累积不透明度
            C=C+1.0*(1-tmp)   # 如果tmp=0，说明应该是背景的颜色（白色）

        return C,D,weights


    @staticmethod
    def sample_pdf(bins,weights,N_importance,det,epsilon=1e-5):
        '''
        bins是前面的t_mids扩展到所有光线
        t_mids等于(t[1:]+t[:-1])/2  形状(N_samples-1,)   存储了所有栅栏区间的中点
        bins  # (N_rays,N_samples-1)
        det=True表示确定性采样
        weights  # (N_rays, N_samples) 是粗采样得到的权重
        '''
        # weights+=epsilon   # 防止除以0
        weights=weights+epsilon  # ⭐不能用上面的原位操作！原位操作会导致梯度图错误，反向传播的时候会报错
        N_rays,N_samples=weights.shape

        pdf=weights/(weights.sum(dim=1,keepdim=True))    # 归一化的概率密度函数
        # pdf (N_rays, N_samples)
        cdf=torch.cumsum(pdf,dim=1)  # 分布函数cdf (N_rays, N_samples)
        cdf_start=torch.zeros_like(cdf[:,:1])   # (N_rays,1)
        cdf=torch.cat([cdf_start,cdf],dim=1)  # 分布函数cdf (N_rays, N_samples+1)


        if det:   # 测试模式要均匀采样
            u=torch.linspace(0.0,1.0,N_importance,device=bins.device)
            u=u.expand([N_rays,N_importance])
        else:     # 训练模式要随机采样
            u=torch.rand([N_rays,N_importance],device=bins.device)
        # u形状(N_rays,N_importance)，取值[0.0, 1.0]



        # cdf形状(N_rays,N_samples+1)，取值[0.0, 1.0],递增。索引从0到64
        # bins形状(N_rays,N_samples-1)，取值(0.0, 6.0)，递增。索引从0到62
        idx=torch.searchsorted(cdf,u,right=True)  # (N_rays,N_importance)
        # 会返回cdf中“首个大于u的元素的索引”
        # idx形状(N_rays,N_importance)，取值[1,65]


        lower=torch.clamp(idx-1, min=0)  # 取值[0,64]
        upper=torch.clamp(idx,   max=N_samples)  # 取值[1,64]
        idx_cdf=torch.stack([lower,upper],dim=-1)
        # (N_rays,N_importance,2)  # 取值[0,64]


        # ⭐ 与cdf_i对应的bins索引是bins_{i-1}
        lower=torch.clamp(idx-2, min=0, max=N_samples-2)  # 取值[0,62]
        upper=torch.clamp(idx-1, min=0, max=N_samples-2)  # 取值[1,62]
        idx_bins=torch.stack([lower,upper],dim=-1)
        # (N_rays,N_importance,2)  # 取值[0,62]




        cdf_expanded=cdf.unsqueeze(dim=1).expand([-1,N_importance,-1]) #(N_rays, N_importance, N_samples+1)
        bins_expanded=bins.unsqueeze(dim=1).expand([-1,N_importance,-1])  #(N_rays, N_importance, N_samples-1)

        cdf_gathered=torch.gather(cdf_expanded,dim=2,index=idx_cdf)  # (N_rays,N_importance,2)
        bins_gathered=torch.gather(bins_expanded,dim=2,index=idx_bins)  # (N_rays,N_importance,2)

        # 线性插值
        # bins -> cdf 这个映射关系是一个随机变量的分布函数
        cdf_qujian_len=cdf_gathered[:,:,1]-cdf_gathered[:,:,0]  # (N_rays,N_importance)
        cdf_qujian_len=torch.clamp(cdf_qujian_len,min=epsilon) # 防止除以0
        
        tau_fine=(u-cdf_gathered[:,:,0])/cdf_qujian_len  # (N_rays,N_importance)
        bins_len=bins_gathered[:,:,1]-bins_gathered[:,:,0]  # (N_rays,N_importance)
        t_fine=bins_gathered[:,:,0]+tau_fine*bins_len  # (N_rays,N_importance)

        return t_fine





    def render(self, batch):
        """
        Write your codes here.
        """
        rays_o=batch['rays_o']  # batch是通过Dataset.getitem得到的字典
        rays_d=batch['rays_d']
        rgbs=batch['rgbs']
        # 三个的形状都是(B,N_rays, 3)

        rays_o=rays_o.flatten(0,1)  # (BN_rays,3)
        rays_d=rays_d.flatten(0,1)  # (BN_rays,3)
        rgbs=rgbs.flatten(0,1)  # (BN_rays,3)

        # 把BN_rays重新定义为N_rays
        N_rays=rays_o.shape[0]
        pts,t_coarse=self._sample_rays(rays_o,rays_d,
                                       self.N_samples,self.perturb,self.lindisp)
        # (N_rays,N_samples,3)    (N_rays,N_samples)
        # pts是光线上的粗采样点



        N_rays,N_samples,_=pts.shape


        if self.double_chunk_coarse:
            # net里面本身会以self.chunk对光线进行分块处理，避免OOM错误
            # 但是所有光线上的所有采样点传入net之后，在embed步骤就会触发OOM了
            # 所以必须双重分块。在传入net之前就手动分块一次
            raw_list=[]
            for i in range(0,N_rays,self.pre_chunk_size):
                pts_chunk=pts[i:i+self.pre_chunk_size]  # (chunk,N_samples,3)
                rays_d_chunk=rays_d[i:i+self.pre_chunk_size]  # (chunk,3)
                # 取出了chunk条光线送入net，避免embed的时候OOM
                raw_chunk=self.net(pts_chunk,rays_d_chunk,model="coarse")
                # (chunk,N_samples,4)
                raw_list.append(raw_chunk)
            raw=torch.cat(raw_list,dim=0).reshape(N_rays,N_samples,4)
            # (N_rays,N_sample,4)，得到了每条光线、每个采样点对应的σ和c
        else:
            raw=self.net(pts,rays_d,model="coarse")
        
        
        C,D,weights=self._raw2output(raw,t_coarse,rays_d,white_bkgd=self.white_bkgd)
        # 这个C,D是粗采样结果
        ret={"C_coarse":C,
             "D_coarse":D,}
        if self.N_importance<=0:   # 说明不需要细采样
            return ret

        # 细采样
        t_mids=(t_coarse[:,:-1]+t_coarse[:,1:])/2  # (N_rays,N_samples-1)
        t_fine=self.sample_pdf(t_mids,weights,self.N_importance,det=not(self.perturb))
        # (N_rays,N_importance)



        # 最终渲染
        t_combine=torch.cat([t_coarse,t_fine],dim=1)  # (N_rays,N_samples+N_importance)
        t_combine,_=torch.sort(t_combine,dim=1)  # 混在一起的所有采样点必须升序排序，才能再喂给网络

        pts_fine=rays_o[:,None]+t_combine.unsqueeze(dim=-1)*rays_d[:,None]
        # 所有光线共享同一组深度值，但是每条光线的rays_d向量不同
        # rays_o[:,None]是(N_rays, 1, 3)
        # pts_fine是最终采样点的三维坐标，形状(N_rays,N_samples+N_importance,3)



        N_rays,N_sam_imp,_=pts_fine.shape        



        if self.double_chunk_fine:
            raw_fine_list=[]
            for i in range(0,N_rays,self.pre_chunk_size):
                pts_chunk=pts_fine[i:i+self.pre_chunk_size]  # (chunk,N_sam_imp,3)
                rays_d_chunk=rays_d[i:i+self.pre_chunk_size]  # (chunk,3)
                # 取出了chunk条光线送入net，避免embed的时候OOM
                raw_chunk=self.net(pts_chunk,rays_d_chunk,model="fine")
                # (chunk,N_samples,4)
                raw_fine_list.append(raw_chunk)
            raw_fine=torch.cat(raw_fine_list,dim=0).reshape(N_rays,N_sam_imp,4)
            # (N_rays,N_samples+N_importance,4)，得到了每条光线、每个采样点对应的σ和c
        else:
            raw_fine=self.net(pts_fine,rays_d,model="fine")
        
        
        
        C_fine,D_fine,weights_fine=self._raw2output(raw_fine,t_combine,rays_d,white_bkgd=self.white_bkgd)
        ret["C_fine"]=C_fine
        ret["D_fine"]=D_fine
        ret["weights_fine"]=weights_fine

        return ret


