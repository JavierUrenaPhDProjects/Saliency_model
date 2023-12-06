import numpy as np

from toolbox.config import load_args
from toolbox.dataloader import create_dataloader, create_dataset
from toolbox.models_utils import load_model, save_model, evaluation_normal_classificator
from toolbox.utils import set_seed, start_logging, save_log, get_batch_loss
from torch import nn, autocast
from torcheval.metrics import MulticlassAUROC
from tqdm import tqdm
import torch.optim as optim


# Definition of training steps:

def step(x, gt, model, optimizer, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    outputs = model(x)
    loss = loss_fn(outputs, gt)
    loss.backward()
    optimizer.step()
    return loss


def AMP_step(x, gt, model, optimizer, loss_fn, scaler):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type='cuda'):
        outputs = model(x)
        loss = loss_fn(outputs, gt)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss


# Training script:

def train(model, train_loader, val_loader, loss_fn, metric, optimizer, scheduler, args, output_batches=True,
          nBatchesOutput=100, max_tries=10):
    device = args['device']
    epochs = args['epochs']

    AMP_flag = False
    if device != 'cpu':
        import torch
        AMP_flag = True
        scaler = torch.cuda.amp.GradScaler()

    print('\n_____Model pre-evaluation_____')
    best_score = evaluation_normal_classificator(model, val_loader, metric)
    print(f'Pre-evaluation score: {best_score}')
    tries = 0

    for epoch in range(epochs):
        print(f'\nCurrent Learning Rate of the training: {optimizer.param_groups[0]["lr"]}')
        model.train()
        running_loss = 0.0
        loss_array = []

        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            if AMP_flag:
                loss = AMP_step(images, labels, model, optimizer, loss_fn, scaler)
            else:
                loss = step(images, labels, model, optimizer, loss_fn)

            loss_value = loss.detach().item()
            running_loss += loss_value
            loss_array = np.append(loss_array, loss_value)

            if output_batches and i % nBatchesOutput == nBatchesOutput - 1:
                avg_loss = running_loss / (nBatchesOutput - 1)
                print(f'Epoch {epoch + 1}, batches {i - (nBatchesOutput - 1)} '
                      f'to {i} average loss (BCE): {avg_loss}')
                # zero the loss
                get_batch_loss(args, [avg_loss])
                running_loss = 0.0

        eval_score = evaluation_normal_classificator(model, val_loader, metric)
        print(f'Epoch {epoch + 1}, | Area Under ROC {eval_score}')
        save_log(args, epoch, loss_array, eval_score)

        if eval_score > best_score:
            save_model(model, args)
            best_score = eval_score
            tries = 0
        else:
            tries += 1
            scheduler.step()
        if tries > max_tries:
            break


if __name__ == '__main__':
    args = load_args('training_classificator')
    set_seed()

    dataset = create_dataset(args)

    train_loader, val_loader = create_dataloader(dataset, args)

    model = load_model(model_name=args['model'], args=args)

    loss_fn = nn.CrossEntropyLoss()
    metric = MulticlassAUROC(num_classes=257)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args['lr'], max_lr=0.1)

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
          f'\nNumber of epochs: {args["epochs"]}')
    print('---------------------')

    start_logging(args)
    train(model, train_loader, val_loader, loss_fn, metric, optimizer, scheduler, args)
