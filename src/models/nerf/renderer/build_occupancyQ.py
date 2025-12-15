import numpy as np
import torch
import torch.nn.functional as F
from src.config import cfg
import os


@torch.no_grad()  # 禁用梯度追踪，减少内存占用
def build_occupancy(net):
    occ_threshold= cfg.task_arg.occ_threshold
    occ_filename= cfg.task_arg.occ_filename
    voxel_chunk=cfg.task_arg.voxel_chunk  # True表示分块处理体素来构建occupancy grid，避免OOM
    voxel_chunk_size=int(cfg.task_arg.voxel_chunk_size)

    occ_res= torch.tensor(cfg.task_arg.occ_res, device=net.device)  
    # tensor([128,128,128])  
    AABB_min= torch.tensor(cfg.task_arg.AABB_min, device=net.device)
    AABB_max= torch.tensor(cfg.task_arg.AABB_max, device=net.device)
    voxel_size=(AABB_max-AABB_min)/occ_res.float()  
    # tensor([4.0/128, 4.0/128, 4.0/128])
    N_voxel_sample=torch.tensor(cfg.task_arg.voxel_sample, device=net.device)
    # tensor([3, 3, 3])

    first_voxel_min=AABB_min
    first_voxel_max=AABB_min+voxel_size


    first_voxel_sample_dimlist=[]
    for dim in range(3):
        first_voxel_sample_dimlist.append(
            torch.linspace(first_voxel_min[dim].item(),
                           first_voxel_max[dim].item(),
                           N_voxel_sample[dim].item(),
                           device=net.device)  # 形状(N)
        )
    first_voxel_meshgrid=torch.meshgrid(*first_voxel_sample_dimlist,indexing="ij")
    # 返回3个均为(N_voxel_sample[0],N_voxel_sample[1],N_voxel_sample=[2])的tensor
    # 即：3个(3,3,3)的矩阵
    first_voxel_samples=torch.stack(first_voxel_meshgrid,dim=-1)
    # (3,3,3,3)
    first_voxel_samples=first_voxel_samples.flatten(0,2)
    # (27,3)  是一个体素内的27个点的坐标

    num_voxels=int(occ_res[0]*occ_res[1]*occ_res[2])  # 总的体素个数 128^3
    N_per_voxel=first_voxel_samples.shape[0]  # 每个体素内的采样点数 27


    ranges=[]
    for dim in range(3):
        ranges.append(
            torch.arange(occ_res[dim].item(),device=net.device)
        )
    index_meshgrid=torch.meshgrid(*ranges,indexing="ij")
    index_coord=torch.stack(index_meshgrid,dim=-1)
    # (128,128,128,3)
    shift=index_coord * voxel_size


    tmp=first_voxel_samples.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # (1,1,1,27,3)
    tmp_expanded=tmp.expand(occ_res[0],occ_res[1],occ_res[2],N_per_voxel,3)
    # (128,128,128,27,3)
    samples=tmp_expanded+ shift.unsqueeze(3)
    # (128,128,128,27,3)

    pts=samples.reshape(-1,3).unsqueeze(1)  
    # (N_total, 1, 3) 相当于每个点来自一条光线（光线方向已经无所谓了）
    del samples   # 释放GPU占用
    N_total=pts.shape[0]   # N_total等于num_voxels*N_per_voxel


    if voxel_chunk:
        sigma=np.zeros((N_total,))
        for i in range(0, N_total, voxel_chunk_size):
            start=i
            end=min(i+voxel_chunk_size,N_total)
            real_chunk_size=end-start
            rays_d=torch.ones(real_chunk_size,3,device=net.device)
            # (chunk_size, 3)
            # 只需要获取体素中各个采样点的σ，而σ与光线方向无关，所以可以随便传入一个单位矢量作为rays_d
            raw_chunk=net(pts[start:end],rays_d,model="fine")  
            # 选取训练好的fine模型，获取最大的精度
            # (chunk_size,1, 4)
            del rays_d
            sigma_chunk=F.relu(raw_chunk[:,:,3]) # (chunk_size,1) 保证体密度非负
            del raw_chunk
            sigma_chunk_cpu=sigma_chunk.squeeze(-1).cpu().numpy() # (chunk_size,)
            # 每次得到raw就立刻提取σ并转移到CPU，不存raw。防止GPU上OOM
            sigma[start:end]=sigma_chunk_cpu
        sigma=torch.from_numpy(sigma).to(net.device)   # (N_total,)
        # 移回GPU
    
    else:  # 一般不会进入这个分支。很容易OOM
        rays_d=torch.ones(N_total,3,device=net.device) # (N_total, 3)
        raw=net(pts,rays_d,model="fine")  #(N_total,1,4)
        sigma=F.relu(raw[:,:,3])  #(N_total,1) 保证体密度非负
        sigma=sigma.squeeze(-1)   # (N_total,)

    del pts  # 必须清除。否则会OOM
    sigma_reshape=sigma.reshape(num_voxels,N_per_voxel)
    # (num_voxels,N_per_voxel)
    sigma_max,_=sigma_reshape.max(dim=1)  #(num_voxels,) 存储了每个体素中最大的sigma
    occupancy=(sigma_max > occ_threshold)
    # (num_voxels,) bool型 存储了每个体素是否是被占据的状态
    occupancy=occupancy.reshape(occ_res[0],occ_res[1],occ_res[2])
    # (128,128,128)

    save_dir="data/Q_occ_grid for ESS/"
    os.makedirs(save_dir,exist_ok=True)
    occ_path=os.path.join(save_dir,f"{occ_filename}.pth")

    torch.save(occupancy,occ_path)


    # 下面代码debug用
    occupied_num=occupancy.sum().item()
    occ_ratio=occupied_num/num_voxels  # 理论上应该在0.05到0.2左右


    return occupancy  # 这个返回可以用于调试

