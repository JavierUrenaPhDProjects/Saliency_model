import random
import torch
import os
import numpy as np
import pandas as pd
import cv2
import time


def set_seed(seed=1000):
    """
    Sets the seed for reproducibility

    Args:
        seed (int, optional): Input seed. Defaults to 1000.
    """
    if seed:
        print(f'Setting random seed {seed}')
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cuda_available = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if cuda_available:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        pass


def activate_AMP(device):
    if device != 'cpu':
        AMP_flag = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        AMP_flag = False
        scaler = None

    return AMP_flag, scaler


def preprocess_img(img_dir, channels=3):
    if channels == 1:
        img = cv2.imread(img_dir, 0)
    elif channels == 3:
        img = cv2.imread(img_dir)

    shape_r = 288
    shape_c = 384
    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def start_logging(args):
    model_name = args['model']
    date = args['date']
    dataset = args['dataset']

    model_path = os.path.join("trained_models", model_name)
    logs_path = os.path.join(model_path, "logs")
    log_filename = f"{model_name}_{dataset}_{date}.csv"
    filepath = os.path.join(logs_path, log_filename)
    losspath = os.path.join(logs_path, 'loss.npy')

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    if os.path.exists(filepath):
        os.remove(filepath)
    if os.path.exists(losspath):
        os.remove(losspath)

    log_df = pd.DataFrame({'epoch': [], 'loss': [], 'eval_score': []})
    log_df.to_csv(filepath, index=False)


def get_epoch_log(epoch, loss_array, eval_score):
    log_dict = {'epoch': epoch,
                'loss': np.average(loss_array),
                'eval_score': eval_score}
    return log_dict


def save_log(args, epoch, loss_array, eval_scor):
    log_dict = get_epoch_log(epoch, loss_array, eval_scor)

    model_name = args['model']
    date = args['date']
    dataset = args['dataset']

    logs_path = os.path.join("trained_models", model_name, "logs")
    log_filename = f"{model_name}_{dataset}_{date}.csv"
    filepath = os.path.join(logs_path, log_filename)

    log_df = pd.read_csv(filepath)
    new_log = pd.DataFrame(log_dict, index=[0])
    log_df = log_df._append(new_log, ignore_index=True)
    log_df.to_csv(filepath, index=False)


def get_batch_loss(args, loss):
    model_name = args['model']
    logs_path = os.path.join("trained_models", model_name, "logs")
    filepath = os.path.join(logs_path, 'loss.npy')
    if not os.path.exists(filepath):
        np.save(filepath, [])
    loss_array = np.load(filepath)
    np.save(filepath, np.append(loss_array, loss))


def test_inference(model, img_size=384, dtype=torch.float64):
    with torch.inference_mode():
        time_array = []
        for i in range(100):  # will measure the average time along 100 iterations
            x1 = torch.randn(1, 3, img_size, img_size).to(dtype)  # RGB image
            x2 = torch.randn(1, 1, img_size, img_size).to(dtype)  # Saliency map

            time_init = time.time()
            _ = model(x1, x2)
            time_array = np.append(time_array, time.time() - time_init)
            print(f' Average Inference time over {i} iterations: {np.average(time_array)}')
