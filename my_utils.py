import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import Resize
from PIL import Image
import os
import cv2
from imutils import paths
from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import CE_Loss
import numpy as np
import matplotlib.pyplot as plt
from torch import unsqueeze
trans=transforms.ToTensor()
class MyData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = sorted(os.listdir(self.root_dir))  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = Image.open(img_path).convert('RGB')  # 读取该图片
        raw =np.array(img)
        #img = img.resize((280,250))
        if self.transform:
            img = self.transform(img)
        
        return (img,index,raw)  # 返回该样本和路径

class Mylabel(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.labels = sorted(os.listdir(self.root_dir))  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.labels)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        label_index = self.labels[index]  # 根据索引index获取该图片
        label_path = os.path.join(self.root_dir, label_index)  # 获取索引为index的图片的路径名
        label=torch.load(label_path)
        return label # 返回该样本和路径

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(3,10,1,stride=1,padding=0,),
                nn.LeakyReLU(),
                nn.BatchNorm2d(9),
                nn.Conv2d(10,10,1)
        )

    def forward(self, x):
        '''
        x: batch, channel, width, height
        '''
        x = self.conv1(x)
        return x


class generator(nn.Module):
    def __init__(self, num_classes,pretrained=True,backbone="mobilenet",downsample_factor=8):
        super().__init__()
        self.generate=DeepLab(num_classes=num_classes,backbone=backbone,pretrained=pretrained,downsample_factor=downsample_factor)
    def forward(self, x):
        x = self.generate.forward(x)
        return x

class painting():
    def __init__(self,batchnum,channel,h,w):
        self.colors=[(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),(0,128,128),(128,128,128),(192,0,0),(64,0,0),(0,0,0)]#rgb
        self.batchnum=batchnum
        self.channel=channel
        self.height=h
        self.width=w
    def paint(self,imgs):
        results=[]
        for i in range(self.batchnum):
            result=np.zeros((500, 560, 3))#h,w,c
            img=imgs[i]
            img=torch.argmax(img,dim=0).cpu().numpy()
            for c in range(self.channel):
                result[:, :, 0] += ((img[:, :] == c ) * self.colors[c][0]).astype('uint8')
                result[:, :, 1] += ((img[:, :] == c ) * self.colors[c][1]).astype('uint8')
                result[:, :, 2] += ((img[:, :] == c ) * self.colors[c][2]).astype('uint8')
            results.append(np.uint8(result))
        return np.hstack(results)

class classify():
    def classifier(self,img,batchnum,channel,high,wide,classes=9):#输入batch_size，h，w，c 输出batch_size，h，w tensor
        result=np.zeros((batchnum,high,wide))
        color={(128,0,0):0,(0,128,0):1,(128,128,0):2,(0,0,128):3,(128,0,128):4,(0,128,128):5,(128,128,128):6,(192,0,0):7,(64,0,0):8,(0,0,0):9}
        for n in range(batchnum):
            single_img=img[n]
            # input_tensor = single_img.clone().detach().to(torch.device('cpu'))# 到cpu
            # single_img = input_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            for h in range(high):
                for w in range(wide):
                    g=tuple(single_img[h,w,:])
                    c=color.get(g,None)
                    result[n][h][w]=c
        return torch.from_numpy(result).long()
    def reverse_classifier(self,img,batchnum,channel,h,w):
        img=img.cpu().numpy()
        result=torch.zeros(batchnum,channel,h,w)
        color={(128,0,0):3,(0,128,0):4,(128,128,0):5,(0,0,128):0,(128,0,128):4,(0,128,128):5,(128,128,128):6,(192,0,0):7,(64,0,0):8,(0,0,0):9}
        for n in range(batchnum):
            single_img=img[n]
            for h in range(h):
                for w in range(w):
                    result[n][color.get(single_img[:,h,w],None)][h][w]=1
        return result

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
class checkpoint():
    def __init__(self,net,optimizer,epoch):
        self.net=net,
        self.optimizer=optimizer,
        self.epoch=epoch
    def save_checkpoint(self):
        if not os.path.isdir("./checkpoints"):
            os.mkdir("./checkpoints")
        torch.save(checkpoint, './models/checkpoint/ckpt_best_%s.pth' %(str(self.epoch)))

def mean_iou(input, target, classes = 2):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    real_classes=0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        if np.sum(union):
            temp = np.sum(intersection) / np.sum(union)
            real_classes+=1
        else:temp=0
        miou += temp
    return  miou/real_classes
# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(hist,gt_img, pred_img, num_classes, name_classes,step,steps):  

    #------------------------------------------------#
    #   对一张图片计算classes×classes的hist矩阵，并累加
    #------------------------------------------------#
    hist += fast_hist(gt_img.flatten(), pred_img.flatten(),num_classes)  
      
    if (step+1) % 100 == 0:
        print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(step+1, steps,
                                                100 * np.nanmean(per_class_iu(hist)),
                                                100 * np.nanmean(per_class_PA(hist))))    
    return hist
