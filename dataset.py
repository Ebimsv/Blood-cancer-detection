import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class CNMC(Dataset):
    def __init__(self, mode, split_size, root_dir, csv_path, transform=None):
        self.mode = mode
        self.split_size = split_size
        self.root_dir = root_dir
        self.csv_path = csv_path
        if transform is None:
            raise ValueError('arg `transform` must be given!')
        self.transform = transform
        self.paths, self.targets = self._load()
        if mode == 'train':
            self.paths = self.paths[:int(len(self.paths) * self.split_size)]
            self.targets = self.targets[:int(len(self.targets) * self.split_size)]
        elif mode == 'test':
            self.paths = self.paths[-int(len(self.paths) * self.split_size):]
            self.targets = self.targets[-int(len(self.targets) * self.split_size):]
        else:
            raise ValueError(f'Invalid `mode`: {mode}. expected either `train` or `test`!')

    def _load(self):
        df = pd.read_csv(self.csv_path, index_col=False)
        targets = df['labels']
        paths = df['new_names'].apply(lambda x: os.path.join(self.root_dir, x))
        return paths.to_numpy(), targets.to_numpy()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path, target = self.paths[index], self.targets[index]
        target = torch.LongTensor([target])
        image = Image.open(path)
        image = self.transform(image)
        return image, target


if __name__ == '__main__':
    import torch.nn as nn
    from model import MedCNN
    from torchvision.models import resnet18
    transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomApply([
            T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        T.RandomGrayscale(0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    d = CNMC('train',
             .8,
             './data/C-NMC_test_prelim_phase_data',
             'data/C-NMC_test_prelim_phase_data_labels.csv',
             transform
             )
    criterion = nn.CrossEntropyLoss()
    sample = d[0][0].unsqueeze(0)
    target = d[0][1]
    model_ = MedCNN(resnet18(), n_class=2)
    output = model_(sample)
    loss_val = criterion(output, target)
    print(d[0])
