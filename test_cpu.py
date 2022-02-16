# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import numpy as np
import os
import scipy.io
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from PIL import Image
from model import ft_net
import numpy as np
import shutil
img_to_tensor = transforms.ToTensor()

######################################################################
# Options
parser = argparse.ArgumentParser(description='Testing')
#parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--query_dir',default='./query/',type=str, help='要识别图像位置')
parser.add_argument('--query_deal_dir',default='./query_deal/',type=str, help='已经识别的图像的位置')
parser.add_argument('--cam_dir',default='./camera/',type=str, help='8个摄像头采集到的图像')
parser.add_argument('--frame_num', default=16,type=int, help='一共要识别多少帧')

opt = parser.parse_args()
#str_ids = opt.gpu_ids.split(',')
which_epoch = 'last'
query_path = opt.query_dir
cam_path = opt.cam_dir
query_deal_path = opt.query_deal_dir
frame_num = opt.frame_num
######################################################################
# set gpu ids
#gpu_ids = []
#for str_id in str_ids:
    #id = int(str_id)
    #if id >=0:
        #gpu_ids.append(id)
#if len(gpu_ids)>0:
    #torch.cuda.set_device(gpu_ids[0])
######################################################################
# Load model
def load_network(network):
    network.load_state_dict(torch.load('./net_last.pth'))
    return network

model_structure = ft_net(751)
model = load_network(model_structure)
# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()
# Change to test mode
model = model.eval()
#use_gpu = torch.cuda.is_available()
#if use_gpu:
    #model = model.cuda()
######################################################################
# Load Data
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

######################################################################
# Extract feature
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_feature(img):
    img = Image.open(img)
    img = data_transforms(img)

    features = torch.FloatTensor()
    img = img.unsqueeze(0)
    n, c, h, w = img.size()

    ff = torch.FloatTensor(n,2048).zero_()

    for i in range(2):
        if(i==1):
            img = fliplr(img)
        #input_img = Variable(img.cuda())
        input_img = Variable(img)
        outputs = model(input_img)
        f = outputs.data
        #f = outputs.data.cpu()
        ff = ff+f
    # norm feature
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))

    features = torch.cat((features,ff), 0)

    return features

def result_cam():
    for query_name in os.listdir(query_path):
        since = time.time()
        query_name_ = query_name.split('.jpg')[0]
        f = open("./res/{}.txt".format(query_name_), "w")
        print("person:{}".format(query_name_))
        print("person:{}".format(query_name_), file=f)
        #检测图片的特征
        query_feature = extract_feature(query_path + query_name)
        #判断是否被检测过
        flag = 0
        for query_deal_img in os.listdir(query_deal_path):
            query_deal_img = query_deal_path + query_deal_img
            query_deal_feature = extract_feature(query_deal_img).permute(1,0)
            temp = np.dot(query_feature, query_deal_feature)
            if temp > 0.92 :
               print("has been detected:{}".format(query_deal_img))
               print("has been detected:{}".format(query_deal_img), file=f)


               flag=1
               shutil.move(query_path + query_name, query_deal_path + query_name)
               break

        if flag == 1:
            continue
        #没有被检测过则开始检测
        #i表示从第1帧到16帧
        for frame in range(1, frame_num+1):
            cam_id = 'cam0'
            cam_time = frame
            max_cam = 0
            #便利每个文件夹下第i帧的图片
            for cam_name in os.listdir(cam_path):
                for num in range(1, 1000):
                    img = cam_path + cam_name + '/' + str(frame) + '_{}.jpg'.format(num)
                    if not os.path.exists(img):
                        break

                    cam_img_feature  = extract_feature(img).permute(1,0)
                    temp = np.dot(query_feature, cam_img_feature)
                    if max_cam < temp:
                        max_img_path = img
                        max_cam = temp
                        cam_id = cam_name
                        cam_time = frame

            if max_cam > 0.92:
                print("frame:" + str(cam_time) +":cam:" + cam_id + ":path:" + max_img_path)
                print("frame:" + str(cam_time) +":cam:" + cam_id + ":path:" + max_img_path, file = f)
            else:
                print("frame:{}:cam:NotFind:path:null".format(cam_time))
                print("frame:{}:cam:NotFind:path:null".format(cam_time), file=f)

        if not os.path.exists(query_deal_path):
            os.makedirs(query_deal_path)
            
        shutil.move(query_path + query_name, query_deal_path + query_name)
        f.close()
        end = time.time()
        print('time:', (end - since))

def detect():
    if os.listdir(query_path):
        result_cam()

if __name__ == '__main__':
    while 1:
        detect()


