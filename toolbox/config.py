import argparse
import os
import nni
from nni.utils import merge_parameter
import logging


def str2bool(v):
    # Función para interpretar palabras booleanas en consola
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def select_dataset(dataset):
    if dataset == 'CALTECH256':
        datasetpath = 'CALTECH256'

    else:
        raise Exception("The name of the Dataset selected is not available")

    return datasetpath


parser = argparse.ArgumentParser('Training arguments')
parser.add_argument('--dataset', default='CALTECH256', type=str)
dataset = parser.parse_known_args()[0].dataset

if 'c703i' in os.uname()[1]:
    data_path = f'/data/javier.urena/{select_dataset(dataset)}/'
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=1000, type=int)

else:
    data_path = f'/home/javier/Pycharm/DATASETS/{select_dataset(dataset)}/'
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--seed', default='', type=str)

# DATASET PARAMETERS #
parser.add_argument('--dataset_path', default=data_path, type=str)

# TRAINING PARAMETERS #
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--train_percnt', default=0.8, type=float)
parser.add_argument('--epochs', default=5, type=int)

# MODEL PARAMETERS #
parser.add_argument('--model', default='SalClass_crossmod_mode1', type=str)
parser.add_argument('--img_size', default=384, type=int)
parser.add_argument('--pretrain', default=False, type=str2bool)
parser.add_argument('--dropout', default=0.0001, type=float)
parser.add_argument('--model_checkpoint', default='last_trained_SalClass_crossmod_mode1_CALTECH256.pth', type=str)
return_args = parser.parse_args()


def load_args(logger_name):
    logger = logging.getLogger(logger_name)
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    args = vars(merge_parameter(return_args, tuner_params))

    return args
