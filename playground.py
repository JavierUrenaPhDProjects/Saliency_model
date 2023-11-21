from models.TranSalNet.TranSalNet_Res import TranSalNet_Res
from toolbox.dataloader import *
from models.SalClass.SalClass_BruteFussion import SalClass

args = {'dataset': 'CALTECH256',
        'img_size': (384, 384),
        'dataset_path': '/home/javier/Pycharm/DATASETS/CALTECH256',
        'batch_size': 2,
        'train_percnt': 0.8,
        'device': 'cpu'}

dataset = create_dataset(args)
train_loader, val_loader = create_dataloader(dataset, args)

img, sal, y, og = dataset.__getitem__(6969)
img, sal = img.unsqueeze(0), sal.unsqueeze(0)

model = SalClass(img_size=args['img_size'])

y = model(img, sal)

# cnn(x)
# cnn_img = medium_cnn(n_channels=3)
# cnn_sal = medium_cnn(n_channels=1)

# y_img = cnn_img(img)
# y_sal = cnn_sal(sal_output)

toPIL = transforms.ToPILImage()
# show saliency map
# pic = np.array(toPIL(sal_output.squeeze()))
# plt.figure()
# plt.imshow(pic)
# plt.show()

# show original image
# img_ = np.array(toPIL(img.squeeze(0)))
# plt.figure()
# plt.imshow(img_)
# plt.show()
