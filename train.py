import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from model import MedCNN
from dataset import CNMC
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchsampler import ImbalancedDatasetSampler

import numpy as np
import random
from tqdm import tqdm
from torchmetrics import Accuracy, F1, Recall, Precision

from config import config
from metrics import AverageMeter

# configs
root_dir = config['root_dir']
train_dir = config['train_dir']
test_dir = config['test_dir']
csv_path = config['csv_path']
img_width = config['img_width']
img_height = config['img_height']
device = config['device']
batch_size = config['batch_size']
epochs = config['epochs']
save_interval = config['save_interval']
weights_dir = config['weights_dir']
seed = config['seed']

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# transforms
transform_train = T.Compose([
    T.Resize((img_height, img_width)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.Resize((img_height, img_width)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# datasets
train_set = ImageFolder(root=train_dir, transform=transform_train)
test_set = ImageFolder(root=test_dir, transform=transform_test)

# dataloaders
sampler = ImbalancedDatasetSampler(train_set)
train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

# model
backbone = models.resnet50(pretrained=True)
model = MedCNN(backbone=backbone, n_class=2)
model.to(device)

# sample = torch.randn(1, 3, 224, 224)
# model(sample)

# optim & criterion
# optimizer = optim.SGD(model.parameters(), lr=config['lr'],
#                       momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

# Trackers
accuracy = Accuracy()
accuracy.to(device)
F1 = F1(num_classes=2, average='macro')
F1.to(device)
recall = Recall(num_classes=2, average='macro')
recall.to(device)
acc_tracker = AverageMeter(name='acc')
loss_tracker = AverageMeter(name='loss')
f1_tracker = AverageMeter(name='f1')
recall_tracker = AverageMeter(name='recall')

# Tensorboard
writer = SummaryWriter('logs')

# training loop
for epoch in range(1, epochs):
    print()
    # train
    model.train()
    running_loss = 0.
    acc_tracker.reset()
    loss_tracker.reset()
    f1_tracker.reset()
    recall_tracker.reset()
    with tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch}/{epochs} ',
              bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as tqdm_dataloader:
        for data in tqdm_dataloader:
            data = [d.to(device) for d in data]
            images, targets = data

            optimizer.zero_grad()
            # forward
            outputs = model(images)
            # backward
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            acc = accuracy(outputs.argmax(1), targets)
            f1 = F1(outputs.argmax(1), targets)
            recall_ = recall(outputs.argmax(1), targets)

            loss_tracker.update(loss.item())
            acc_tracker.update(acc.item())
            f1_tracker.update(f1.item())
            recall_tracker.update(recall_.item())

            avg_train_loss = loss_tracker.avg
            avg_train_acc = acc_tracker.avg
            avg_train_f1 = f1_tracker.avg
            avg_train_recall = recall_tracker.avg

            # tqdm_dataloader.set_postfix(loss=loss.item(), acc=acc.item(), f1=f1.item(), recall=recall_.item())

            tqdm_dataloader.set_postfix(loss=avg_train_loss, acc=avg_train_acc, f1=avg_train_f1, recall=avg_train_recall)

    # eval
    model.eval()
    loss_tracker.reset()
    acc_tracker.reset()
    f1_tracker.reset()
    recall_tracker.reset()
    with tqdm(test_loader, unit="batch", desc=f'Evaluating... ',
              bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as tqdm_dataloader:
        for data in tqdm_dataloader:
            with torch.no_grad():
                data = [d.to(device) for d in data]
                images, targets = data
                outputs = model(images)

                test_loss = criterion(outputs, targets)
                test_acc = accuracy(outputs.argmax(1), targets)
                test_f1 = F1(outputs.argmax(1), targets)
                test_recall = recall(outputs.argmax(1), targets)

                loss_tracker.update(test_loss.item())
                acc_tracker.update(test_acc.item())
                f1_tracker.update(test_f1.item())
                recall_tracker.update(test_recall.item())

                avg_test_loss = loss_tracker.avg
                avg_test_acc = acc_tracker.avg
                avg_test_f1 = f1_tracker.avg
                avg_test_recall = recall_tracker.avg

                # tqdm_dataloader.set_postfix(loss=test_loss.item(), acc=test_acc.item(), f1=test_f1.item(),
                #                             recall=test_recall.item())

                tqdm_dataloader.set_postfix(loss=avg_test_loss, acc=avg_test_acc, f1=avg_test_f1, recall=avg_test_recall)

    scheduler.step()

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Val', avg_test_loss, epoch)
    writer.add_scalar('Acc/Train', avg_train_acc, epoch)
    writer.add_scalar('Acc/Val', avg_test_acc, epoch)

    if epoch % save_interval == 0:
        checkpoint = {'state_dict': model.state_dict(),
                      'epoch': epoch,
                      'optimizer': optimizer.state_dict()}
        save_path = f'{weights_dir}/{epoch}.pt'
        torch.save(checkpoint, save_path)
