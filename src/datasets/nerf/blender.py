import torch.utils.data as data
import torch
import numpy as np
from src.config import cfg

#下面头文件是自己加的
import os
import json
from PIL import Image


def _get_rays(H,W,K,c2w):
    i,j=torch.meshgrid(torch.linspace(0,W-1,W),   #(W,) -> (W,H)  i是W方向的索引
                       torch.linspace(0,H-1,H),   #(H,) -> (W,H)  j是H方向的索引
                       indexing="ij")
    i=i.t()  #(H,W)
    j=j.t()  #(H,W)

    fx=K[0,0]
    fy=K[1,1]
    cx=K[0,2]
    cy=K[1,2]

    d_cam=torch.stack([(i-cx)/fx,
                       -(j-cy)/fy,       #对于图片，在H方向，索引是向下增大的。要取负
                       -torch.ones_like(i)],  #Z=-1是归一化平面
                       dim=-1)
    #(H,W,3)



    #c2w是(4,4)，可以分成
    # (R,t)
    # (0,1)  
           
    R=c2w[:3,:3]  #(3,3)
    t=c2w[:3,-1]  #(3,)

    rays_d=d_cam @ R.t()  #(H,W,3)  这里需要对R转置，才能得到正确的旋转方向
    rays_o=t.expand([H,W,3])  #(H,W,3)  这是相机在真实世界的坐标，要扩展到(H,W,3)

    rays_o=rays_o.reshape(-1,3)  #(HW,3)
    rays_d=rays_d.reshape(-1,3)  #(HW,3)

    return rays_o,rays_d


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
        Output:
            None
        """
        super(Dataset, self).__init__()
        """
        Write your codes here.
        """
        self.split=kwargs.get("split","train")

        if self.split=="train":
            self.root=os.path.join(cfg.train_dataset.data_root,cfg.scene)   #cfg.scene会是"lego"
            self.is_train=True
            self.n_rays=cfg.task_arg.N_rays
            self.ratio=cfg.train_dataset.input_ratio
            self.H=int((cfg.train_dataset.H)*self.ratio)
            self.W=int((cfg.train_dataset.W)*self.ratio)
            self.cams=cfg.train_dataset.cams   # [0,-1,1]
            
        else:
            self.root=os.path.join(cfg.test_dataset.data_root,cfg.scene)   #cfg.scene会是"lego"
            self.is_train=False
            self.n_rays=-1
            self.ratio=cfg.test_dataset.input_ratio
            self.H=int((cfg.test_dataset.H)*self.ratio)
            self.W=int((cfg.test_dataset.W)*self.ratio)
            self.cams=cfg.test_dataset.cams   # [0,-1,100]

        json_path=os.path.join(self.root,f"transforms_{self.split}.json")
        with open(json_path,'r') as f:
            meta=json.load(f)
    
        self.frames=meta["frames"]
        a,b,c=self.cams
        self.frames=self.frames[a:b:c]  #按照cfg中的cams进行切片
        self.len=len(self.frames) #这是图片的总数N

        camera_angle_x=meta["camera_angle_x"]
        f=0.5*self.W/(np.tan(0.5*camera_angle_x))
        cx=self.W/2
        cy=self.H/2

        self.K=torch.tensor([[f,0,cx],
                             [0,f,cy],
                             [0,0,1]],dtype=torch.float32)  #❓如果用np创建矩阵，再转化成torch tensor?
        
        all_rays_o=[]
        all_rays_d=[]
        all_rgbs=[]



        for frame in self.frames:
            img_path=os.path.join(self.root,frame["file_path"]+'.png')
            img=Image.open(img_path)  #这是一个PIL.Image.Image 类型的对象
            # print(img.mode)   #应当是RGBA

            if self.ratio!=1:
                img=img.resize([self.W,self.H],resample=Image.Resampling.LANCZOS)

            img=np.array(img).astype(np.float32)/255.0   #形状(H,W,4)，每个元素是[0.0,1.0]之间的浮点数
            rgb=img[...,:3]*img[...,-1:]+1.0*(1-img[...,-1:])  #alpha混合，背景是白色1.0
            rgb=torch.from_numpy(rgb.reshape(-1,3)) #(H,W,3)->(HW,3)

            c2w=torch.tensor(frame["transform_matrix"]).float()  #(4,4)
            rays_o,rays_d=_get_rays(self.H,self.W,self.K,c2w)  #两个都是(HW,3)
            # 一张图片中所有HW个像素对应的一共HW条光线，每条光线有一个长度3的位置向量o和长度3的方向向量d

            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)
            all_rgbs.append(rgb)

        #训练模式，需要从所有图片的所有光线里面随机挑1024条
        self.rays_o=torch.cat(all_rays_o,dim=0)  #(NHW,3)   #这里拼接了所有图片的所有像素对应的光线
        self.rays_d=torch.cat(all_rays_d,dim=0)  #(NHW,3)
        self.rgbs=torch.cat(all_rgbs,dim=0)  #(NHW,3)

        if self.is_train==False:   #测试模式，需要选取特定索引的图片中的所有光线
            self.rays_o=self.rays_o.reshape(self.len,self.H,self.W,3)  #(N,H,W,3)
            self.rays_d=self.rays_d.reshape(self.len,self.H,self.W,3)  #(N,H,W,3)
            self.rgbs=self.rgbs.reshape(self.len,self.H,self.W,3)  #(N,H,W,3)



    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典
        """
        """
        Write your codes here.
        """

        num_rays_total=self.rays_o.shape[0]  #训练模式下是NHW

        if self.is_train:
            sample_idx=torch.randint(0,num_rays_total,size=(self.n_rays,))   #从所有图片的所有像素对应的总共NHW条光线里面随机选1024条
            rays_o=self.rays_o[sample_idx]  #(1024,3)
            rays_d=self.rays_d[sample_idx]  #(1024,3)
            rgbs=self.rgbs[sample_idx]  #(1024,3)
            return {
                "rays_o":rays_o,
                "rays_d":rays_d,
                "rgbs":rgbs
                }     # 这里必须返回字典，不能用元组。因为后面训练的框架期望从Dataset的getitem得到的就是字典

        else:   #测试模式
            rays_o=self.rays_o[index]  #(H,W,3)
            rays_d=self.rays_d[index]  #(H,W,3)
            rgbs=self.rgbs[index]  #(H,W,3)
            return {
                "rays_o":rays_o.reshape(-1,3),  #(HW,3)
                "rays_d":rays_d.reshape(-1,3),  #(HW,3)
                "rgbs":rgbs.reshape(-1,3)  #(HW,3)
                }
            



    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量（即可迭代次数。对于train模式而言是NHW条光线大杂烩，对于test模式是N张图片）

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        """
        Write your codes here.
        """
        if self.is_train:
            return self.rays_o.shape[0]  #就是NHW。因为训练的时候是从所有图片的所有一共NHW条光线里面随机抽1024条
        else:
            return self.len  #就是N。因为测试的时候是对于指定的一张图片分析里面的所有光线
