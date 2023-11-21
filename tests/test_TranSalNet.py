import os
import torch
import cv2
import numpy as np
from torchvision import transforms, utils, models
import torch.nn as nn
from tqdm import tqdm
from models.TranSalNet.utils.data_process import preprocess_img, postprocess_img
from PIL import Image

from models.TranSalNet.TranSalNet_Res import TranSalNet_Res
from models.TranSalNet.TranSalNet_Dense import TranSalNet_Dense

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LOAD OF TRANSALNET WITH RESNET BACKBONE
model_res = TranSalNet_Res({'device': 'cpu'})

# LOAD OF TRANSALNET WITH DENSE CONNECTION NETWORK BACKBONE
model_dense = TranSalNet_Dense({'device': 'cpu'})

models = [model_res, model_dense]  # LIST OF MODELS

# TEST IMAGE PREPROCESSING
test_img = r'/home/javier/Pycharm/DATASETS/CALTECH256/256_ObjectCategories/001.ak47/001_0021.jpg'
img = preprocess_img(test_img)  # padding and resizing input image into 384x288
img = np.array(img) / 255.
img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
img = torch.from_numpy(img)
img = img.type(torch.FloatTensor).to(device)

img_n = 1

for model in models:
    model = model.to(device)
    model.eval()

    pred_saliency = model(img)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())

    pred_saliency = postprocess_img(pic, test_img)  # restore the image to its original size as the result

    cv2.imwrite(f'/home/javier/Pycharm/PycharmProjects/Saliency_model/test_data/result_{img_n}.jpg', pred_saliency,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # save the result
    print('Finished, check the result at: {}'.format(r'test_data/result.png'))
    img_n += 1
