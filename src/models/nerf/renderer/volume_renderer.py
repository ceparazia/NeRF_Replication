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
        self.batch_size = net.batch_size  # 也就是cfg.task_arg.N_rays，是1024
        self.white_bkgd = net.white_bkgd
        self.use_viewdirs = net.use_viewdirs
        self.device = net.device

        self.chunk=cfg.task_arg.rays_chunk_size   # 这是喂给MLP的光线的分块大小
        self.perturb= cfg.task_arg.perturb
        self.lindisp= cfg.task_arg.lindisp
        
        self.use_ERT= cfg.task_arg.use_ERT   # ❓ 应该设置成训练模式禁用ERT，测试模式使用ERT
        self.ERT_threshold= cfg.task_arg.ERT_threshold
        self.ERT_chunk=cfg.task_arg.ERT_chunk
        if self.use_ERT:
            print(f"启用ERT. ERT_threshold={self.ERT_threshold}")


        self.use_ESS= cfg.task_arg.use_ESS
        self.force_gen_occ_grid=cfg.task_arg.force_gen_occ_grid
        self.occ_grid=None
        if self.use_ESS:
            occ_filename=cfg.task_arg.occ_filename
            save_dir="data/Q_occ_grid for ESS/"
            occ_path=os.path.join(save_dir,f"{occ_filename}.pth")
            if self.force_gen_occ_grid:   # 强制重新生成网格
                print(f"正在生成{occ_filename}.pth")
                build_occupancyQ.build_occupancy(net)
                try:
                    self.occ_grid=torch.load(occ_path).to(self.device)
                    print(f"生成完毕。已启用ESS，加载{occ_filename}.pth")
                except FileNotFoundError:
                    print(f"生成失败。禁用ESS")
            else:
                try:
                    self.occ_grid=torch.load(occ_path).to(self.device)
                    print(f"启用ESS.加载已有的occupancy grid:{occ_filename}.pth")
                except FileNotFoundError:
                    print(f"尝试启用ESS。但是未找到{occ_filename}.pth")
                    print(f"正在生成{occ_filename}.pth")
                    build_occupancyQ.build_occupancy(net)

                    try:
                        self.occ_grid=torch.load(occ_path).to(self.device)
                        print(f"生成完毕。已启用ESS，加载{occ_filename}.pth")
                    except FileNotFoundError:
                        print(f"生成失败。禁用ESS")

            if self.occ_grid==None:
                self.use_ESS=False

            self.perturb=False # ESS必须均匀采样，而不是分层采样
            self.occ_grid=self.occ_grid.permute(2,1,0)  # 因为F.grid_sample是Z,Y,X维度。

        self.t_near=getattr(cfg.task_arg,"t_near",2.0)
        self.t_far=getattr(cfg.task_arg,"t_far",6.0)  # ❓我在cfg里没看到这两个设定。AI说一般取0.0和6.0



    def get_mask(self,pts):
        AABB_min= torch.tensor(cfg.task_arg.AABB_min, device=self.device)
        AABB_max= torch.tensor(cfg.task_arg.AABB_max, device=self.device)
        pts_norm=2*(pts-AABB_min)/(AABB_max-AABB_min)-1  # 世界坐标 → [-1.0,1.0]
        grid=self.occ_grid
        grid=grid.unsqueeze(0).unsqueeze(0)  # (1,1,128,128,128)

        N_rays,N_samples,_=pts.shape
        pts_norm_flat=pts_norm.reshape(1,1,1,-1,3)

        occ_val=F.grid_sample(
            grid.float(),
            pts_norm_flat,
            mode="nearest",  # 最近邻插值最快且最准确
            padding_mode="zeros",   # 凡是超出 AABB 范围的点(pts_norm值落在[-1,1]之外)，都认为是空 (0)
            align_corners=False  # 
        )

        mask=(occ_val.reshape(N_rays,N_samples)==1)
        # print(mask.sum())   值为2528072，远小于160000*64
        # mask.sum(dim=1)   观察每条光线上的有效采样点。最大的是59
        # mask.sum()/mask.shape[0]  平均每条光线上有15.800个有效采样点
        return mask


    def net_batched(self,pts,rays_d,model="coarse"):
        '''
        整合了chunking和ESS的MLP查询接口
        '''
        N_rays,N_samples,_=pts.shape  
        # ⭐这里的N_rays,N_samples是由输入的pts形状决定的，不是全局量
        chunk=self.chunk
        raw=torch.zeros(N_rays,N_samples,4,device=self.device)
        pts_flat=pts.flatten(0,1)
        rays_d_flat=rays_d.unsqueeze(1).expand(N_rays,N_samples,3).flatten(0,1)  # (N_rays*N_samples ,  3)
        raw_flat=raw.flatten(0,1)   # (N_rays*N_samples ,  4)

        if self.use_ESS:
            mask=self.get_mask(pts)  # (N_rays, N_samples)
            mask_flat=mask.flatten(0,1)
            if not mask.any():
                return raw  # 全是废点，则返回全0的raw
            valid_idx=torch.nonzero(mask_flat).squeeze()
            pts_valid=pts_flat[valid_idx]  # (M,3)   M是所有的有效点个数
            rays_d_valid=rays_d_flat[valid_idx]  # (M,3)
        else:
            valid_idx=torch.arange(pts_flat.shape[0],device=self.device)
            pts_valid=pts_flat
            rays_d_valid=rays_d_flat
        # 形状已经统一
        # pts_valid     (M,3)   后面还需.unsqueeze(1)才能喂给MLP
        # rays_d_valid  (M,3)

        M=pts_valid.shape[0]
        raw_list=[]
        for i in range(0,M,chunk):
            pts_chunk=(pts_valid[i:i+chunk]).unsqueeze(1)   # (chunk,1,3)
            rays_d_chunk=rays_d_valid[i:i+chunk]            # (chunk,3)
            raw_list.append(self.net(pts_chunk,rays_d_chunk,model=model))  # (chunk,1,4)
        raw_slice=torch.cat(raw_list,dim=0)  # (M,1,4)
        
        raw_flat[valid_idx]=raw_slice.squeeze(1)  # 回填
        raw=raw_flat.reshape(N_rays,N_samples,4)
        return raw

            
            



    # 在光线的近平面t_near和远平面t_far之间生成N_samples个采样点。
    # r(t)=o+t*d
    def _sample_rays(self,rays_o,rays_d,N_samples,
                     perturb=True,lindisp=False):   #默认进行分层随机采样、深度均匀采样
        # rays_o,rays_d都是(N_rays,3)
        N_rays=rays_o.shape[0]
        valid_mask=None

        tau=torch.linspace(0.0 , 1.0, N_samples,device=self.device)
        if lindisp==True:
            tmp=tau*(1/self.t_far)+(1-tau)*(1/self.t_near)
            t=1/tmp
        else:
            t=tau*self.t_far+(1-tau)*self.t_near
        # 共(N_samples,)个点，从t_far均匀到t_near均匀分布，“栅栏区间”
        # 如果perturb==False，那么这已经是采样结果了

        if perturb>0:
            mids=(t[1:]+t[:-1])/2   #(N_samples-1,)   存储了所有栅栏区间的中点
            upper=torch.cat([mids,t[-1:]],dim=-1)  # (N_samples,)
            lower=torch.cat([t[:1],mids],dim=-1) # (N_samples,)

            #在每个区间 [lower, upper] 内随机均匀采样
            tau2=torch.rand(N_samples,device=self.device)  # 0.0到1.0之间
            t=lower+tau2*(upper-lower)  # (N_samples,)

        t=t.expand([N_rays,N_samples])  # (N_rays,N_samples)
        # 所有光线共享同一组深度值，但是每条光线的rays_d向量不同
        # pts是采样点的三维坐标，形状(N_rays,N_samples,3)
        pts=rays_o[:,None]+t.unsqueeze(dim=-1)*rays_d[:,None]
        
        if self.use_ESS:
            valid_mask=self.get_mask(pts)

            
        return pts,t,valid_mask
        # (N_rays,N_samples,3)    (N_rays,N_samples)  (N_rays,N_samples)




    #将NeRF预测的原始值σ，c转化为最终的像素颜色C(r)和深度D(r)。
    def _raw2output(self,raw,t,rays_d,white_bkgd=False):
        '''
        仅用于Coarse阶段。进行一次性的渲染
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

        C=(weights[:,:,None]*c).sum(dim=1)  # 最终的像素颜色C(r)    (N_rays,3)
        D=(weights*t).sum(dim=-1)  # depthmap D(r)    (N_rays)

        if white_bkgd:
            tmp=weights.sum(dim=-1,keepdim=True)  # (N_rays,1) 是每条光线的累积不透明度
            C=C+1.0*(1-tmp)   # 如果tmp=0，说明应该是背景的颜色（白色）

        return C,D,weights



    # 增量式的体渲染。用于ERT
    def _ERT_raw2output(self,T_ray,raw_slice,t_slice,deltas_slice,rays_d_slice,white_bkgd=False):
        '''
        T_ray 传入的所有活着的光线的实时能量  (N_active,1)
        raw_slice       (N_active,ERT_chunk,4)
        t_slice         (N_active,ERT_chunk)
        deltas_slice    (N_active,ERT_chunk)
        rays_d_slice    (N_active,3)
        '''
        N_active,ERT_chunk=t_slice.shape
        
        c_slice=torch.sigmoid(raw_slice[:,:,:3])  # (N_rays, N_samples, 3) 保证值在[0,1]之间
        sigma_slice=F.relu(raw_slice[:,:,3])  #(N_rays, N_samples) 保证体密度非负

        tmp=torch.exp(-sigma_slice*deltas_slice)  #(N_rays, N_samples)
        tmp_start=torch.ones([N_active,1],device=self.device)
        tmp=torch.cat([tmp_start,tmp[:,:-1]],dim=-1)   # (N_rays, N_samples)
        T_decay=torch.cumprod(tmp,dim=-1)  # T就是“到达这个点还有多少光”
        # (N_active, ERT_chunk)
        T_ray=T_ray*T_decay

        weights_slice=T_ray*(1-torch.exp(-sigma_slice*deltas_slice))  # (N_rays, N_samples)
        # weights越大，表示这个采样点将对最终渲染出来的RGB的贡献越大

        C_slice=(weights_slice[:,:,None]*c_slice).sum(dim=1)  # 最终的像素颜色C(r)    (N_rays,3)
        D_slice=(weights_slice*t_slice).sum(dim=-1)  # depthmap D(r)    (N_rays)

        if white_bkgd:
            tmp=weights_slice.sum(dim=-1,keepdim=True)  # (N_rays,1) 是每条光线的累积不透明度
            C_slice=C_slice+1.0*(1-tmp)   # 如果tmp=0，说明应该是背景的颜色（白色）

        T_ray=T_ray[:,:-1]  # 每条光线只需取最后一个采样点的T
        return T_ray,C_slice,D_slice,weights_slice



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
        pts,t_coarse,valid_mask=self._sample_rays(rays_o,rays_d,
                                       self.N_samples,self.perturb,self.lindisp)
        # (N_rays,N_samples,3)    (N_rays,N_samples)  (N_rays,N_samples)
        # pts是光线上的粗采样点



        N_rays,N_samples,_=pts.shape

        raw=self.net_batched(pts,rays_d,model="coarse")
        # 我已经把ESS的判断、光线的分块都整合在net_batched方法里面了

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

        # 最终采样深度
        t_combine=torch.cat([t_coarse,t_fine],dim=1)  # (N_rays,N_samples+N_importance)
        t_combine,_=torch.sort(t_combine,dim=1)  # 混在一起的所有采样点必须升序排序，才能再喂给网络

        pts_fine=rays_o[:,None]+t_combine.unsqueeze(dim=-1)*rays_d[:,None]
        # 所有光线共享同一组深度值，但是每条光线的rays_d向量不同
        # rays_o[:,None]是(N_rays, 1, 3)
        # pts_fine是最终采样点的三维坐标，形状(N_rays,N_samples+N_importance,3)

        N_rays,N_combine,_=pts_fine.shape 
        
        deltas=t_combine[:,1:]-t_combine[:,:-1]  #深度区间的差
        deltas=torch.cat([deltas,1e10*torch.ones_like(t_combine[:,:1])],dim=-1)  #(N_rays,N_sam_imp)
        # 最远的那个区间，认为区间距离是无限大
        rays_d_norm=torch.norm(rays_d[:,None,:],dim=-1)  # (N_rays,1)
        # 乘上d的模长才是真正的空间距离
        deltas=deltas*rays_d_norm # (N_rays,N_sam_imp)


        T_ray=torch.ones(N_rays,1,device=self.device)  # ⭐每一批次，实时记录所有光线的剩余能量
        active_mask=torch.ones(N_rays,dtype=torch.bool,device=self.device)
        ERT_chunk=cfg.task_arg.ERT_chunk

        C_fine=torch.zeros(N_rays,3,device=self.device)
        D_fine=torch.zeros(N_rays,device=self.device)
        weights_fine=torch.zeros(N_rays,N_combine,device=self.device)

        for i in range(0,N_combine,ERT_chunk):
            if not active_mask.any():
                break   # 所有光线都已经耗尽了 
            pts_chunk=pts_fine[:,i:i+ERT_chunk]     # (N_rays,ERT_chunk,3)
            deltas_chunk=deltas[:,i:i+ERT_chunk]    # (N_rays,ERT_chunk)
            t_chunk=t_combine[:,i:i+ERT_chunk]      # (N_rays,ERT_chunk)
            raw_chunk=torch.zeros(N_rays,pts_chunk.shape[1],4,device=self.device)
            
            pts_active=pts_chunk[active_mask]       # (N_active,ERT_chunk,3)
            rays_d_active=rays_d[active_mask]       # (N_active,ERT_chunk,3)
            if pts_active.shape[0]>0:   # 说明仍然有光线活着
                raw_active= self.net_batched(pts_active,rays_d_active,model="fine")
                raw_chunk[active_mask]=raw_active
            c_chunk=torch.sigmoid(raw_chunk[...,:3])
            sigma_chunk=F.relu(raw_chunk[...,3])
            # 下面进行的是逐步的体渲染
            tmp_chunk=torch.exp(-sigma_chunk*deltas_chunk)
            tmp_start=torch.ones([N_rays,1],device=self.device)
            tmp_chunk=torch.cat([tmp_start,tmp_chunk],dim=-1)   # (N_rays, 1+ERT_chunk)
            tmp2=torch.cumprod(tmp_chunk,dim=-1)  # (N_rays, 1+ERT_chunk)，开头是1
            T_local=tmp2[:,:-1]    # (N_rays, ERT_chunk)
            T_ray_decay=tmp2[:,-1:]  # (N_rays, 1)

            weights_chunk=T_ray*T_local*(1-torch.exp(-sigma_chunk*deltas_chunk))  # (N_rays, N_samples)

            C_fine=C_fine+(weights_chunk[:,:,None]*c_chunk).sum(dim=1)  # 最终的像素颜色C(r)    (N_rays,3)
            D_fine=D_fine+(weights_chunk*t_chunk).sum(dim=-1)  # depthmap D(r)    (N_rays)
            weights_fine[:,i:i+ERT_chunk]=weights_chunk

            T_ray=T_ray*T_ray_decay  # ⭐这里更新了T_ray
            if self.use_ERT:
                alive=(T_ray>self.ERT_threshold).squeeze(-1)
                active_mask=active_mask&alive

        if self.white_bkgd:
            C_fine=C_fine+1.0*T_ray       
                
        ret["C_fine"]=C_fine
        ret["D_fine"]=D_fine
        ret["weights_fine"]=weights_fine
        return ret


