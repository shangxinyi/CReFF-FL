import torch
import random
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image


class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label


class ImageNetLTDataLoader(DataLoader):
    def __init__(self, shuffle=True, training=True):
        data_dir = os.path.dirname(__file__)
        data_dir = os.path.dirname(__file__) + '/ImageNet_LT/'
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if training:
            dataset = LT_Dataset(data_dir, data_dir + 'ImageNet_LT_train.txt', train_trsfm)
            val_dataset = LT_Dataset(data_dir, data_dir + 'ImageNet_LT_val.txt', train_trsfm)
        else:  # test
            dataset = LT_Dataset(data_dir, data_dir + 'ImageNet_LT_test.txt', test_trsfm)
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)
        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 1000
        self.num_classes = num_classes
        self.list_label2indices = self.classifity_index_label()

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list
        self.shuffle = shuffle
        self.init_kwargs = {
            'shuffle': self.shuffle
        }
        super().__init__(dataset=self.dataset, **self.init_kwargs)

    def split_validation(self):
        # If you do not want to validate:
        #return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, shuffle=True)

    def classifity_index_label(self):
        list_label2indices = [[] for _ in range(self.num_classes)]
        for idx, label in enumerate(self.dataset):
            list_label2indices[label].append(idx)

