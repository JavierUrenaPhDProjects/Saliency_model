import os
from models.TranSalNet.TranSalNet_Res import TranSalNet_Res
import torch
import numpy as np
import cv2
from torchvision import transforms
from models.TranSalNet.utils.data_process import preprocess_img, postprocess_img
from models.TranSalNet.TranSalNet_Res import TranSalNet_Res
from tqdm import tqdm

root_dir = '/home/javier/Pycharm/DATASETS/CALTECH256/'
input_dir = os.path.join(root_dir, '256_ObjectCategories')
output_dir = os.path.join(root_dir, 'saliencies')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_res = TranSalNet_Res({'device': 'cpu'})
model_res.to(device)
model_res.eval()
toPIL = transforms.ToPILImage()


def input_output(in_dir, out_dir, filename):
    input = os.path.join(in_dir, filename)
    output = os.path.join(out_dir, filename)
    return input, output


for cls_folder in tqdm(os.listdir(input_dir)):
    print(f'Creating saliencies of folder {cls_folder}')
    input_folder, output_folder = input_output(input_dir, output_dir, cls_folder)
    try:
        os.mkdir(output_folder)
    except:
        pass

    for img_file in os.listdir(input_folder):
        input_file, output_file = input_output(input_folder, output_folder, img_file)
        if not os.path.exists(output_file) and img_file.endswith(('.jpg', '.jpeg', '.png')):
            img = preprocess_img(input_file)
            img = np.array(img) / 255.
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            img = torch.from_numpy(img)

            sal = model_res(img)
            sal = toPIL(sal.squeeze())
            sal = postprocess_img(sal, input_file)
            cv2.imwrite(output_file, sal, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
