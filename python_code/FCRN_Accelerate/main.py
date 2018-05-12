# coding=utf-8

import numpy as np
import os
from skimage import img_as_float, io, transform
import time
from FCRN_Accelerate.model import *
from FCRN_Accelerate.weights import *
import torch
import cv2 as cv
from torch.autograd import Variable
from torchvision.utils import save_image
import sys

def image_center_extract(image, new_c, new_r):
    print(image.shape)
    y, x = image.shape[:2]
    print(x, y)
    start_c = x // 2 - (new_c // 2)
    start_r = y // 2 - (new_r // 2)

    return image[start_r : start_r+new_r, start_c : start_c+new_c, :]

if __name__ == '__main__':

    #define FCRN model and load the model of FCRN from NYU_ResNet-UpProj.npy
    model = FCRN().cuda()
    model.load_state_dict(load_weights(model, 'NYU_ResNet-UpProj.npy', torch.cuda.FloatTensor))

    #load image and convert image to size(304, 228)
    img = img_as_float(io.imread("1.ppm"))
    #cv.imwrite('in.png', img)
    img2 = img_as_float(io.imread('1.ppm'))
    net_img = torch.from_numpy(image_center_extract(img, 304, 228)).permute(2,0,1).unsqueeze(0).float().cuda()
    #print(net_img.type())
    save_image(net_img.cpu(), "in.png")
    # print(net_img.size())
    #build calculation graph
    fcrn_input = Variable(net_img)

    start_t = time.time()
    #print(fcrn_input.type())
    out_img = model(fcrn_input).cuda()
    end_t = time.time()
    print("Finish time is: %f", end_t -  start_t)
    out = out_img[0]

    out = (out.cpu().detach().numpy())
    out = (out - out.min()) / (out.max()-out.min())*255

    # out = out.astype('uint8')
    out = np.transpose(out, (1, 2, 0))
    out = cv.resize(out, (304, 228))
    #out = transform.resize(out, (1242, 375))

    cv.imwrite('out.png', out)
    #save_image(out, "out.png", normalize = True)

