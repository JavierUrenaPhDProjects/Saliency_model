import os
import torch.cuda
from models.SalClass.SalClass_BruteFussion import *
from models.SalClass.SalClass_EmbedFusion import *
from models.SalClass.SalClass_CrossModal import *
from models.SalClass.parts.vision_transformer import vit_base
from tqdm import tqdm
import datetime


def load_model(model_name, args):
    model = eval(model_name + f'({args})')
    if args['pretrain']:
        if model_name in args['model_checkpoint']:
            ckpt_file = args['model_checkpoint']
        else:
            ckpt_file = f'last_trained_{model_name}_{args["dataset"]}.pth'

        print(f'Loading pre-trained model of {model_name}. Checkpoint: {ckpt_file}')

        try:
            checkpoint = torch.load(
                f'./trained_models/{model_name}/{ckpt_file}',
                map_location=torch.device(args['device']))
            model.load_state_dict(checkpoint, strict=False)
            model.to(args['device'])
            print(f"Model {model_name} loaded")
        except:
            print(f'File {ckpt_file} not found')
            print(f'Checkpoint for model: {model_name}, trained in dataset: {args["dataset"]} not found!')
            print('The model will be loaded FROM SCRATCH')

    model.to(args['dtype'])
    print(f'Size of the architecture: {sum(p.numel() for p in model.parameters())} parameters')

    model.to(args['device'])
    print("The model will be running on", args['device'], "device")

    return model


def save_model(model, args):
    print('_____New best model encountered! saving checkpoint_____')
    hostname = os.uname()[1]
    today = datetime.date.today().strftime("%d-%m-%Y")
    name = args['model']
    dataset = args['dataset']
    path = f"trained_models/{name}"

    if not os.path.exists(path):
        os.mkdir(path)

    torch.save(model.state_dict(), f"{path}/{name}_{dataset}_{hostname}_{today}.pth")
    torch.save(model.state_dict(), f"{path}/last_trained_{name}_{dataset}.pth")


def evaluation(model, val_loader, loss_fn, metric):
    print('Evaluating model...')
    metric.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    cum_loss = 0

    with torch.no_grad():
        for images, saliencies, labels in tqdm(val_loader):
            # Prepare data and give it to the model
            images, saliencies, labels = images.to(device), saliencies.to(device), labels.to(device)
            pred = model(images, saliencies)
            # Measure the loss and metric from the predictions
            loss = loss_fn(pred, labels)
            cum_loss += loss.detach().item()
            labels_parsed = labels.argmax(dim=1)
            metric.update(pred, labels_parsed)

    val_score = metric.compute()
    val_loss = cum_loss / len(val_loader)
    return val_score.detach().item(), val_loss


def evaluation_normal_classificator(model, val_loader, loss_fn, metric):
    print('Evaluating model...')
    metric.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    cum_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            pred = model(images)

            loss = loss_fn(pred, labels)
            cum_loss += loss.detach().item()
            labels_parsed = labels.argmax(dim=1)
            metric.update(pred, labels_parsed)

    val_score = metric.compute()
    val_loss = cum_loss / len(val_loader)
    return val_score.detach().item(), val_loss
