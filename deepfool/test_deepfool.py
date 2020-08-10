import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os


# 获得一个提前训练好的model
#https://github.com/MyRespect/AdversarialAttack
net = models.resnet34(pretrained=True)

# Switch to evaluation mode
# 测试模型
net.eval()

# 输入图片
im_orig = Image.open('test_im1.jpg')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]


# Remove the mean
im = transforms.Compose([
    transforms.Scale(256), #指定大小
    transforms.CenterCrop(224),#中心切割224大小图片
    transforms.ToTensor(), #转换为张量
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)  #正则化

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

# tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
#                         transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
#                         transforms.Lambda(clip),
#                         transforms.ToPILImage(),
#                         transforms.CenterCrop(224)])

tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
transforms.Lambda(clip),
transforms.ToPILImage(),
transforms.CenterCrop(224)])

plt.figure()
plt.imshow(tf(pert_image.cpu()[0]))
plt.title(str_label_pert)
plt.show()
