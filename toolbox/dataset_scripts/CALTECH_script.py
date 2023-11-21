import pandas as pd
import os

datasetpath = '/home/javier/Pycharm/DATASETS/CALTECH256/'
imagespath = os.path.join(datasetpath, '256_ObjectCategories')
outfile = 'CALTECH256_dataframe.csv'

dic = {'label': [],
       'id': [],
       'imgfile': []
       }

for cls in os.listdir(imagespath):
    cls_folder = os.path.join(datasetpath, cls)
    for img_file in os.listdir(cls_folder):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            id, label = cls.split('.')
            dic['label'].append(label)
            dic['id'].append(id)
            dic['file'].append(os.path.join(cls, img_file))

df = pd.DataFrame(dic)
df = df.sort_values(by=['id'])
df.to_csv(os.path.join('/home/javier/Pycharm/DATASETS/CALTECH256/', outfile), index=False)
