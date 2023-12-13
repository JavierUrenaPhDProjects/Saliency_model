import numpy as np

from toolbox.config import load_args
from toolbox.dataloader import create_dataloader, create_dataset
from toolbox.models_utils import load_model, save_model, evaluation
from toolbox.utils import set_seed, start_logging, save_log, get_batch_loss, activate_AMP
from torch import nn, autocast
from torcheval.metrics import MulticlassAUROC
from tqdm import tqdm
import torch.optim as optim


# Definition of training steps:

# Normal training step
def step(x1, x2, gt, model, optimizer, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    outputs = model(x1, x2)
    loss = loss_fn(outputs, gt)
    loss.backward()
    optimizer.step()
    return loss


# Automatic Mixed Precission training step
def AMP_step(x1, x2, gt, model, optimizer, loss_fn, scaler):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type='cuda'):
        outputs = model(x1, x2)
        loss = loss_fn(outputs, gt)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss


# Configure learning rate scheduler:
def lr_scheduler(optimizer, args, data_len, mode='triangular2', n_cycles=5):
    if args['lr_scheduler']:
        n_epochs = args['epochs']
        batch_size = args['batch_size']
        data_size = int(data_len * args["train_percnt"])

        iter_per_epoch = data_size // batch_size
        max_iter = n_epochs * iter_per_epoch

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args['lr'], max_lr=args['lr'] * 4, mode=mode,
                                                step_size_up=max_iter // (n_cycles * 2))
    else:
        scheduler = None
    return scheduler


# Training script:

def train(model, train_loader, val_loader, loss_fn, metric, optimizer, scheduler, args, output_batches=True,
          nBatchesOutput=100, patience=10):
    device = args['device']
    epochs = args['epochs']

    AMP_flag, scaler = activate_AMP(device)

    print('\n_____Model pre-evaluation_____')
    best_score = evaluation(model, val_loader, metric)
    print(f'\nPre-evaluation score: {best_score}')
    tries = 0

    for epoch in range(epochs):
        print(f'\nCurrent Learning Rate of the training: {optimizer.param_groups[0]["lr"]}')
        model.train()
        running_loss = 0.0
        loss_array = []

        for i, (images, saliencies, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, saliencies, labels = images.to(device), saliencies.to(device), labels.to(device)

            if AMP_flag:
                loss = AMP_step(images, saliencies, labels, model, optimizer, loss_fn, scaler)
            else:
                loss = step(images, saliencies, labels, model, optimizer, loss_fn)

            loss_value = loss.detach().item()
            running_loss += loss_value
            loss_array = np.append(loss_array, loss_value)

            if args['lr_scheduler']:
                scheduler.step()

            if output_batches and i % nBatchesOutput == nBatchesOutput - 1:
                avg_loss = running_loss / (nBatchesOutput - 1)
                print(f'Epoch {epoch + 1}, batches {i - (nBatchesOutput - 1)} '
                      f'to {i} average loss (Cross Entropy Loss): {avg_loss}')
                # zero the loss
                get_batch_loss(args, [avg_loss])
                running_loss = 0.0

        eval_score = evaluation(model, val_loader, metric)
        print(f'Epoch {epoch + 1}, | Area Under ROC {eval_score}')
        save_log(args, epoch, loss_array, eval_score)

        if eval_score > best_score:
            save_model(model, args)
            best_score = eval_score
            tries = 0
        else:
            tries += 1
        if tries > patience:
            break


if __name__ == '__main__':
    args = load_args('training_classificator')
    set_seed()

    dataset = create_dataset(args)

    train_loader, val_loader = create_dataloader(dataset, args)

    model = load_model(model_name=args['model'], args=args)

    loss_fn = nn.CrossEntropyLoss()
    metric = MulticlassAUROC(num_classes=dataset.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay'])

    scheduler = lr_scheduler(optimizer, args, dataset.__len__())

    print('\n---------------------\nTraining model\n---------------------')
    print(f'Model: {args["model"]}'
          f'\nPre-trained: {args["pretrain"]}')
    print('\n---------------------\nDataset information\n---------------------')
    print(f'Training model on dataset: {args["dataset"]}'
          f'\nDataset size: {dataset.__len__()}'
          f'\nData dimensions: {dataset.__getitem__(0)[0].shape}'
          f'\nData type: {dataset.__getitem__(0)[0].dtype}')
    print('\n---------------------\nTraining parameters\n---------------------')
    print(f'Train size: {int(dataset.__len__() * args["train_percnt"])}'
          f'\nValidation size: {int(dataset.__len__() - dataset.__len__() * args["train_percnt"])}'
          f'\nBatch size: {args["batch_size"]}'
          f'\nLearning rate: {args["lr"]}'
          f'\nNumber of epochs: {args["epochs"]}'
          f'\nPatience: {int(args["epochs"] * 0.1)}'
          )
    print('---------------------')

    start_logging(args)
    train(model, train_loader, val_loader, loss_fn, metric, optimizer, scheduler, args,
          patience=int(args['epochs'] * 0.1))
