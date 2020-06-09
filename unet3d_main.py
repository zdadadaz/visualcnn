# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import transforms
import numpy as np
import cv2
from functools import partial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from models import UNet3D_ef
from models import UNet3D_ef_deconv
from utils import *



def load_images(img_path):
    # imread from img_path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))

    # pytorch must normalize the pic by 
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    img = transform(img)
    img.unsqueeze_(0)
    #img_s = img.numpy()
    #img_s = np.transpose(img_s, (1, 2, 0))
    #cv2.imshow("test img", img_s)
    #cv2.waitKey()
    return img

def store(model):
    """
    make hook for feature map
    """
    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool3d):
           # print(output[0],output[1])
           model.feature_maps[key] = output[0]
           model.pool_locs[key] = output[1]
        else:
           model.feature_maps[key] = output
    
    for idx, layer in enumerate(model._modules.get('features')):    
        # _modules returns an OrderedDict
        layer.register_forward_hook(partial(hook, key=idx))

def vis_layer(layer, vgg16_conv, vgg16_deconv):
    """
    visualing the layer deconv result
    """
    new_img = []
    max_activation = 0
    num_feat = vgg16_conv.feature_maps[layer].to("cpu").detach().numpy().shape[1]
    # set other feature map activations to zero
    # (1, 30, 32, 112, 112)
    new_feat_map = vgg16_conv.feature_maps[layer].clone()

    # # choose the max activations map
    act_lst_time = []
    for t in range(32):
        act_lst = []
        for i in range(0, num_feat):
            choose_map = new_feat_map[0, i,t, :, :]
            activation = torch.max(choose_map)
            act_lst.append(activation.item())
        act_lst_time.append(act_lst)

    act_lst_time = np.array(act_lst_time)
    # print(act_lst_time)
    mark = np.argmax(act_lst_time, axis=1)
    # print(mark)
    # # make zeros for other feature maps
    for t in range(32):
        choose_map = new_feat_map[0, mark[t], t, :, :] # grab the featuremap with the maximum activation value
        max_activation = torch.max(choose_map)
        if mark[t] == 0:
            new_feat_map[:, 1:, :, :] = 0
        else:
            new_feat_map[:, :mark[t],t, :, :] = 0
            if mark[t] != vgg16_conv.feature_maps[layer].shape[1] - 1:
                new_feat_map[:, mark[t] + 1:,t, :, :] = 0
    
        choose_map = torch.where(choose_map==max_activation,
                choose_map,
                torch.zeros(choose_map.shape).to(device)
                )

        # make zeros for other activations
        new_feat_map[0, mark[t],t, :, :] = choose_map
        # print(max_activation.item())
    # print(new_feat_map.size())
    deconv_output = vgg16_deconv(new_feat_map, layer, mark, vgg16_conv.pool_locs)
    # (1, 30, 32, 112, 112) batch,feature,time,w,h
    new_img = deconv_output.data.to("cpu").detach().numpy()[0,:,0,...].transpose(1, 2, 0)  # (H, W, C)
    print(new_img.shape)
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)
    # cv2.imshow('reconstruction img ' + str(layer), new_img)
    # cv2.waitKey()
    return new_img, int(max_activation)
    

if __name__ == '__main__':
    
    img_path = './data/0X1A0A263B22CCD966.avi'
    # torch.cuda.clear_memory_allocated()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # forward processing
    video = loadvideo(img_path)
    img = video[:,:2:-1,:,:]
    img = img[:,:32,...]
    img = img[np.newaxis,...]
    model = UNet3D_ef(in_channels=3, out_channels=1)
    img = torch.from_numpy(img.copy())
    img = img.to(device)
    model.to(device)
    
    model.eval()
    store(model)
    conv_output = model(img)
    pool_locs = model.pool_locs
    # print('Predicted:', decode_predictions(conv_output, top=3)[0])
    

    
    # # backward processing
    model_decon = UNet3D_ef_deconv()
    model_decon.eval()
    # plt.figure(num=None, figsize=(16, 12), dpi=80)
    # plt.subplot(2, 4, 1)
    # plt.title('original picture')
    # img = cv2.imread(img_path)
    # img = cv2.resize(img, (224, 224))
    # plt.imshow(img)    
    
    layer_list = [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]
    for idx, layer in enumerate(layer_list):
        plt.subplot(2, 4, idx+2)
        img, activation = vis_layer(layer, model, model_decon)
        plt.title(f'{layer} layer, the max activations is {activation}')
        # img = img[112,112,:]
        plt.imshow(img)
        plt.colorbar()

    plt.show()
    # plt.savefig('result.jpg')
    # print('result picture has save at ./result.jpg')
