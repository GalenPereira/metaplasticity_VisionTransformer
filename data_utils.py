import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import json
from collections import OrderedDict
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets

import torch.utils.data




from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VisionDataset


'''mnist_dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
mnist_train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=100, shuffle=True, num_workers=1)

mnist_dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=100, shuffle=False, num_workers=1)

fmnist_dset_train = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
fashion_mnist_train_loader = torch.utils.data.DataLoader(fmnist_dset_train, batch_size=100, shuffle=True, num_workers=1)

fmnist_dset_test = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
fashion_mnist_test_loader = torch.utils.data.DataLoader(fmnist_dset_test, batch_size=100, shuffle=False, num_workers=1)
'''
def download_and_unzip(URL, root_dir):
  error_message = "Download is not yet implemented. Please, go to {URL} urself."
  raise NotImplementedError(error_message.format(URL))

def _add_channels(img, total_channels=3):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
 
#usps_transform = torchvision.transforms.Compose( [torchvision.transforms.Resize((28,28)),
#    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

#usps_dset_train = torchvision.datasets.USPS('./usps_pytorch', train=True, transform=usps_transform, target_transform=None, download=True)
#usps_train_loader = torch.utils.data.DataLoader(usps_dset_train, batch_size=12, shuffle=True, num_workers=1)

#usps_dset_test = torchvision.datasets.USPS('./usps_pytorch', train=False, transform=usps_transform, target_transform=None, download=True)
#usps_test_loader = torch.utils.data.DataLoader(usps_dset_test, batch_size=100, shuffle=False, num_workers=1)

class DatasetProcessing(torch.utils.data.Dataset): 
    def __init__(self, data, target, transform=None): 
        self.transform = transform
        self.data = data.astype(np.float32)[:,:,None] if data.ndim == 2 else data
        self.target = torch.from_numpy(target).long()

    def __getitem__(self, index): 
        data = self.data[index]
        if data.ndim == 2:  # for grayscale to match with 3 channel expectation
            data = np.tile(data, (3, 1, 1))
        data = Image.fromarray(data)
        if self.transform is not None:
            data = self.transform(data)
        return data, self.target[index]

    def __len__(self): 
        return len(self.data)

'''# Transform to normalize and resize images to 80x80
transform = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])'''

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def process_features(X_train, X_test, mode):
    if mode=="cutoff":
        cutoff = 8
        threshold_train = np.zeros((np.shape(X_train)[0],1))
        threshold_test = np.zeros((np.shape(X_test)[0],1)) 
        for i in range(np.shape(X_train)[0]):
            threshold_train[i,0] = np.unique(X_train[i,:])[-cutoff]
        for i in range(np.shape(X_test)[0]):
            threshold_test[i,0] = np.unique(X_test[i,:])[-cutoff]
        X_train =   (np.sign(X_train  - threshold_train + 1e-6 ) + 1.0)/2
        X_test =  (np.sign (X_test  - threshold_test +1e-6 ) + 1.0)/2
    elif mode=="mean_over_examples":
        X_train = ( X_train - X_train.mean(axis = 0, keepdims = True) )/ X_train.var(axis =0, keepdims = True) # ???
        X_test = ( X_test - X_test.mean(axis=0, keepdims = True) ) /X_test.var(axis = 0, keepdims = True)
    elif mode=="mean_over_examples_sign":
        X_train =   (np.sign(X_train  - X_train.mean(axis = 0, keepdims = True) ) + 1.0)/2
        X_test =  (np.sign (X_test  - X_test.mean(axis = 0, keepdims = True) ) + 1.0)/2
    elif mode=="mean_over_pixels":
        X_train = ( X_train - X_train.mean(axis = 1, keepdims = True) )/ X_train.var(axis =1, keepdims = True)  # Instance norm
        X_test = ( X_test - X_test.mean(axis=1, keepdims = True) ) /X_test.var(axis = 1, keepdims = True)
    elif mode=="mean_over_pixels_sign":
        X_train =   (np.sign(X_train  - X_train.mean(axis = 1, keepdims = True) ) + 1.0)/2  
        X_test =  (np.sign (X_test  - X_test.mean(axis = 1, keepdims = True) ) + 1.0)/2
    elif mode=="global_mean":
        X_train = ( X_train - X_train.mean(keepdims = True) )/ X_train.var(keepdims = True) # Batch norm
        X_test = ( X_test - X_test.mean(keepdims = True) ) /X_test.var(keepdims = True)
    elif mode=="rescale":
        X_train =  (X_train / X_train.max(axis = 1, keepdims = True) )
        X_test =  (X_test / X_test.max(axis = 1, keepdims = True) )
    return X_train, X_test


def relabel(label):
    label_map = [5,6,0,1,2,3,4,7,8,9]
    return label_map[label]

vrelabel = np.vectorize(relabel)


def process_cifar100(root, n_subset, transform):
    subset_size = 100 // n_subset
    cifar100_train = CIFAR100(root=root, train=True, download=True, transform=transform)
    cifar100_test = CIFAR100(root=root, train=False, download=True, transform=transform)

    train_loader_list, test_loader_list = [], []
    
    for k in range(n_subset):
        indices_train = [i for i, label in enumerate(cifar100_train.targets) if label // subset_size == k]
        indices_test = [i for i, label in enumerate(cifar100_test.targets) if label // subset_size == k]

        train_subset = Subset(cifar100_train, indices_train)
        test_subset = Subset(cifar100_test, indices_test)

        train_loader = DataLoader(train_subset, batch_size=20, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=20, shuffle=False, num_workers=4)

        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)

    return train_loader_list, test_loader_list


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, indices=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load class labels for train and validation.
        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            self.classes = {cls.strip(): idx for idx, cls in enumerate(f.readlines())}

        # Load images and labels
        for cls, idx in self.classes.items():
            cls_folder = os.path.join(root_dir, 'train', cls, 'images')
            for img_name in os.listdir(cls_folder):
                self.images.append(os.path.join(cls_folder, img_name))
                self.labels.append(idx)

        # Apply subset if indices are provided
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def random_split(indices, lengths):
    """
    Randomly split a list of indices into two sets based on the specified lengths.
    """
    np.random.shuffle(indices)
    return indices[:lengths[0]], indices[lengths[0]:]

# Create the data loaders
def create_tiny_imgnet_loaders(root_dir, num_subsets, batch_size, transform):
    full_dataset = TinyImageNetDataset(root_dir, transform=transform)
    total_images = len(full_dataset)
    images_per_subset = total_images // num_subsets

    train_loader_list = []
    test_loader_list = []
    task_names = []

    for i in range(num_subsets):
        subset_indices = list(range(i * images_per_subset, (i + 1) * images_per_subset))
        
        train_size = int(0.8 * len(subset_indices))
        test_size = len(subset_indices) - train_size
        train_indices, test_indices = random_split(subset_indices, [train_size, test_size])

        train_dataset = TinyImageNetDataset(root_dir, transform=transform, indices=train_indices)
        test_dataset = TinyImageNetDataset(root_dir, transform=transform, indices=test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)
        task_names.append(f"Task {i+1}")

    return train_loader_list, test_loader_list, task_names


def createHyperparametersFile(path, args):

    hyperparameters = open(path + r"/hyperparameters.txt","w+")
    L = ["- scenario: {}".format(args.scenario) + "\n",
        "- interleaved: {}".format(args.interleaved) + "\n",
        "- hidden layers: {}".format(args.hidden_layers) + "\n",
        "- normalization: {}".format(args.norm) + "\n",
        "- net: {}".format(args.net) + "\n",
        "- task sequence: {}".format(args.task_sequence) + "\n",
        "- lr: {}".format(args.lr) + "\n",
        "- gamma: {}".format(args.gamma) + "\n",
        "- meta: {}".format(args.meta) + "\n",
        "- beaker: {}".format(args.beaker) + "\n",
        "- number of beakers: {}".format(args.n_bk) + "\n",
        "- ratios: {}".format(args.ratios) + "\n",
        "- areas: {}".format(args.areas) + "\n",
        "- feedback: {}".format(args.fb) + "\n",
        "- ewc: {}".format(args.ewc) + "\n",
        "- ewc lambda: {}".format(args.ewc_lambda) + "\n",
        "- SI: {}".format(args.si) + "\n",
        "- Binary Path Integral: {}".format(args.bin_path) + "\n",
        "- SI lambda: {}".format(args.si_lambda) + "\n",
        "- decay: {}".format(args.decay) + "\n",
        "- epochs per task: {}".format(args.epochs_per_task) + "\n",
        "- init: {}".format(args.init) + "\n",
        "- init width: {}".format(args.init_width) + "\n",
        "- seed: {}".format(args.seed) + "\n"]
   
    hyperparameters.writelines(L)
    hyperparameters.close()
        

