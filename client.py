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
from utils.my_utils import MyData,Mylabel,discriminator,generator,painting
import streamlit as st
from io import BytesIO

classes=10
G = generator(num_classes=classes,pretrained=False,backbone="xception",downsample_factor=8).cuda()
checkpoint_path='D:\Deeplabv3tian\checkpoints\checkpoint-275.pth'
checkpoint=torch.load(checkpoint_path)
G.load_state_dict(checkpoint["net"],strict=True)
G.eval()
trans=transforms.ToTensor()
paint=painting(batchnum=1,channel=classes,h=500,w=560)

st.title('Mars Segmentation')
uploaded_files=st.file_uploader("please choose origin pictures",accept_multiple_files=True)
if uploaded_files is not None:
    with st.expander("See explanation"):
        st.write("""
                The image above shows 9 classes of the Mars surface, black pixels means BACKGROUND. 
                Here is the reference.
            """)
        st.image(Image.open('E:\Marsegdataset-master\pic\counter.png'))
    for i,uploaded_file in enumerate(uploaded_files):
        bytes_data=uploaded_file.getvalue()
        bytes_data=BytesIO(bytes_data)
        img=Image.open(bytes_data).convert('RGB').resize((560,500))
        z=trans(img).cuda()
        z=torch.unsqueeze(z,dim=0)
        with torch.no_grad():
            splited_image=G(z)
        splited_image =paint.paint(splited_image)
        st.image(splited_image,caption='splited_image-{}'.format(i+1))
        splited_image=cv2.cvtColor(splited_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite('splited_img.png',splited_image)
        with open('splited_img.png','rb') as file:   
            btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name='splited_image-{}.png'.format(i+1),
                        mime="image/png"
                    )
        