import numpy as np
from src.config import cfg
import os
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(
        self,
    ):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)   # 这是以dB为单位的信噪比
        # 因为img的rgb值是归一化的，取值∈[0.0, 1.0]，峰值取1
        return psnr   # psnr越大越好，30dB以上是高质量渲染

    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
        result_dir = os.path.join(cfg.result_dir, "images")
        os.system("mkdir -p {}".format(result_dir))
        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            (img_pred[..., [2, 1, 0]] * 255),
        )   # cv2需要的是BGR格式，要从RGB转换一下
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            (img_gt[..., [2, 1, 0]] * 255),
        )
        img_pred = (img_pred * 255).astype(np.uint8)

        # ❓为什么不对img_gt进行转化？
        img_gt = (img_gt * 255).astype(np.uint8)  # ❗ 这一行是我自己加的。可能有错


        ssim = compare_ssim(img_pred, 
                            img_gt, 
                            win_size=101, 
                            full=True,
                            channel_axis=2,  # ❗ 这一个参数是我自己加的。显式地指定颜色轴，防止报错
                            data_range=255)   # ❗ 这一个参数是我自己加的。显式地设置传入的数据范围
        return ssim  # ssim越接近1越好，表示预测图像与真实图像在结构上越相似

    def evaluate(self, output, batch):
        """
        这个方法在每个batch之后都会调用
        output是volume_renderer.py中的render方法返回的字典
        batch是Dataset.getitem的返回值。训练模式是1024条光线，测试模式是1张图片中的所有光线
        """
        img_pred=output["C_fine"]
        img_gt=batch["rgbs"].flatten(0,1)   #(B,N_rays,3) -> (BN_rays,3)
        # B=1。后面把BN_rays看作N_rays

        img_pred=img_pred.detach().cpu().numpy()  # 先切断梯度追踪，再移到CPU，用于后续的numpy操作
        img_gt=img_gt.detach().cpu().numpy()

        self.imgs.append(
            {  
            "img_pred":img_pred,   # (N_rays,3)
            "img_gt":img_gt        # (N_rays,3)
            })



    def summarize(self):   # 这个方法在所有批次评估结束后调用
        """
        Write your codes here.
        """
        if not self.imgs:
            print("出错了")
        
        img_pred_list=[dic["img_pred"]for dic in self.imgs]
        img_gt_list=[dic["img_gt"]for dic in self.imgs]
        
        all_pred=np.concatenate(img_pred_list,axis=0)  # (N_rays_total,3)
        all_gt=np.concatenate(img_gt_list,axis=0)  # (N_rays_total,3)
        N_rays_total=all_pred.shape[0]  # 320000
        
        input_ratio=cfg.test_dataset.input_ratio  # 0.5
        H=int(cfg.test_dataset.H * input_ratio)  # 800*0.5=400
        W=int(cfg.test_dataset.W * input_ratio)  # 800*0.5=400
        N_rays_per_img=int(H*W)   # 160000


        if N_rays_total<N_rays_per_img:
            print("光线不足以重建完整图片")
            return
        
        num_imgs=N_rays_total//N_rays_per_img
        for id in range(num_imgs):
            start=id*N_rays_per_img
            end=(id+1)*N_rays_per_img
            img_pred_1=all_pred[start:end].reshape(H,W,3)
            img_gt_1=all_gt[start:end].reshape(H,W,3)


            img_mse=np.mean((img_pred_1-img_gt_1)**2)   # 这是一个标量
            self.mse.append(img_mse)

            final_psnr=self.psnr_metric(img_pred_1,img_gt_1)
            self.psnr.append(final_psnr)

            batch=1
            ssim_res=self.ssim_metric(img_pred_1,img_gt_1,batch,id,num_imgs)   # 会把预测结果存为第id号图片
            # compare_ssim()的返回值是一个元组
            # ssim_res[0]是一个标量，即为ssim值
            # ssim_res[1]是一个(400,400,3)的矩阵
            final_ssim=ssim_res[0]   
            self.ssim.append(final_ssim)


        avg_mse=np.mean(self.mse)
        avg_psnr=np.mean(self.psnr)
        avg_ssim=np.mean(self.ssim)

        print("===")
        print("评估结果：")
        print("图片总数: ",num_imgs)
        print("Average PSNR(dB): ",avg_psnr)
        print("Average SSIM: ",avg_ssim)
        print("Average MSE: ",avg_mse)
        print("===")





        ret={
            "Average PSNR(dB)": torch.tensor(avg_psnr,dtype=torch.float32),
            "Average SSIM: ": torch.tensor(avg_ssim,dtype=torch.float32),
            "Average MSE: ": torch.tensor(avg_mse,dtype=torch.float32),
            # 显式地转化成0D的变量tensor。如果直接传的话，后续写入tensorboard会报错

            "PSNR_list": torch.tensor(self.psnr,dtype=torch.float32),
            "SSIM_list": torch.tensor(self.ssim,dtype=torch.float32),
            "MSE_list": torch.tensor(self.mse,dtype=torch.float32)
            # 把列表显式地转化成1D的变量tensor，准备写入tensorboard
        }

        self.mse=[]
        self.psnr=[]
        self.ssim=[]
        self.imgs=[]

        return ret