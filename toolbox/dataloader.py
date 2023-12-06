import os.path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader


def pair(n):
    return n if isinstance(n, tuple) else (n, n)


def create_dataset(args):
    datasetName = args['dataset']
    dataset_path = args['dataset_path']
    img_size = pair(args['img_size'])
    dtype = args['dtype']
    if datasetName == 'CALTECH256':
        dataset = CALTECH256(datasetpath=dataset_path, img_size=img_size, dtype=dtype)
    elif datasetName == 'CALTECH256_classification':
        dataset = CALTECH256_classification(datasetpath=dataset_path, img_size=img_size, dtype=dtype)
    return dataset


def create_dataloader(dataset, args):
    trainPercnt = args['train_percnt']
    batchSize = args['batch_size']
    datasetSize = len(dataset)
    trainsize, testsize = int(trainPercnt * datasetSize), datasetSize - int(trainPercnt * datasetSize)

    train_dataset, test_dataset = random_split(dataset, [trainsize, testsize])
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


class CALTECH256(Dataset):
    def __init__(self, datasetpath='/home/javier/Pycharm/DATASETS/CALTECH256',
                 datafile='CALTECH256_dataframe.csv',
                 img_size=(384, 384), dtype=torch.float64):
        self.imagespath = os.path.join(datasetpath, '256_ObjectCategories')
        self.salienciespath = os.path.join(datasetpath, 'saliencies')
        self.datafile = os.path.join(datasetpath, datafile)
        self.img_size = img_size
        self.dataFrame = pd.read_csv(self.datafile)
        self.num_classes = 257
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(img_size, antialias=True),
             # transforms.Normalize(mean=[0.485, 0.456, 0.406],
             #                      std=[0.229, 0.224, 0.225]),
             transforms.ConvertImageDtype(dtype=dtype)
             ])
        self.dtype = dtype

    def __len__(self):
        return len(self.dataFrame)

    def load_image(self, file):
        imgpath = os.path.join(self.imagespath, file)
        salpath = os.path.join(self.salienciespath, file)
        return Image.open(imgpath).convert('RGB'), Image.open(salpath)

    def changetype(self, img, sal):
        # og = np.asarray(img.resize(self.img_size))
        img = np.asarray(img) / 255.
        sal = np.asarray(sal) / 255.
        return img, sal  # , og

    def __getitem__(self, idx):
        _, id, imgFile = self.dataFrame.loc[idx]

        img, sal = self.load_image(imgFile)

        img, sal = self.changetype(img, sal)
        x1, x2 = self.transform(img), self.transform(sal)

        label = np.zeros(self.num_classes)
        label[id - 1] = 1.

        y = torch.tensor(label, dtype=self.dtype)
        return x1, x2, y  # , og


class CALTECH256_classification(Dataset):
    def __init__(self, datasetpath='/home/javier/Pycharm/DATASETS/CALTECH256',
                 datafile='CALTECH256_dataframe.csv',
                 img_size=(384, 384), dtype=torch.float64):
        self.imagespath = os.path.join(datasetpath, '256_ObjectCategories')
        self.datafile = os.path.join(datasetpath, datafile)
        self.img_size = img_size
        self.dataFrame = pd.read_csv(self.datafile)
        self.num_classes = 257
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(img_size, antialias=True),
             # transforms.Normalize(mean=[0.485, 0.456, 0.406],
             #                      std=[0.229, 0.224, 0.225]),
             transforms.ConvertImageDtype(dtype=dtype)
             ])
        self.dtype = dtype

    def __len__(self):
        return len(self.dataFrame)

    def load_image(self, file):
        imgpath = os.path.join(self.imagespath, file)
        return Image.open(imgpath).convert('RGB')

    def changetype(self, img):
        img = np.asarray(img) / 255.
        return img

    def __getitem__(self, idx):
        _, id, imgFile = self.dataFrame.loc[idx]

        img = self.load_image(imgFile)

        img = self.changetype(img)
        x = self.transform(img)

        label = np.zeros(self.num_classes)
        label[id - 1] = 1.

        y = torch.tensor(label, dtype=self.dtype)
        return x, y
