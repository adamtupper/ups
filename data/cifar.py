import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision import datasets, transforms


def get_cifar10(args, splits_dir=".", root='data/datasets', n_lbl=4000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt='', seed=None):
    os.makedirs(root, exist_ok=True)  # Create the root directory for saving data augmentations

    crop_size = 32
    crop_ratio = 0.875
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if args.data_aug == "none":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.data_aug == "weak":
        transform_train = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif args.data_aug == "strong":
        transform_train = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError("Unknown data augmentation setting: {}".format(args.data_aug))
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_idx = np.random.default_rng(seed=seed).choice(50000, size=45000, replace=False)
    val_idx = np.setdiff1d(np.arange(50000), train_idx)
    
    if ssl_idx is None:
        base_dataset = CIFAR10SSL(root, indexs=train_idx, train=True, download=False)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.indexs, base_dataset.targets, n_lbl, 10)
        
        os.makedirs(f'{splits_dir}/data/splits', exist_ok=True)
        f = open(os.path.join(f'{splits_dir}/data/splits', f'cifar10_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    if pseudo_lbl is not None:
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #balance the labeled and unlabeled data 
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    train_lbl_dataset = CIFAR10SSL(
        root,
        lbl_idx,
        train=True,
        transform=transform_train,
        pseudo_idx=pseudo_idx,
        pseudo_target=pseudo_target,
        nl_idx=nl_idx,
        nl_mask=nl_mask
    )
    
    if nl_idx is not None:
        train_nl_dataset = CIFAR10SSL(
            root,
            np.array(nl_idx),
            train=True,
            transform=transform_train,
            pseudo_idx=pseudo_idx,
            pseudo_target=pseudo_target,
            nl_idx=nl_idx,
            nl_mask=nl_mask
        )

    train_unlbl_dataset = CIFAR10SSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    val_dataset = CIFAR10SSL(root, indexs=val_idx, train=True, transform=transform_val, download=False)
    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=False)

    if (nl_idx is not None) and (len(nl_idx) > 0):
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, val_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, val_dataset, test_dataset


def get_cifar100(args, splits_dir=".", root='data/datasets', n_lbl=10000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt='', seed=None):
    crop_size = 32
    crop_ratio = 0.875
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    if args.data_aug == "none":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.data_aug == "weak":
        transform_train = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif args.data_aug == "strong":
        transform_train = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError("Unknown data augmentation setting: {}".format(args.data_aug))
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])
    
    train_idx = np.random.default_rng(seed=seed).choice(50000, size=45000, replace=False)
    val_idx = np.setdiff1d(np.arange(50000), train_idx)

    if ssl_idx is None:
        base_dataset = CIFAR100SSL(root, indexs=train_idx, train=True, download=False)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.indexs, base_dataset.targets, n_lbl, 100)
        
        os.makedirs(f'{splits_dir}/data/splits', exist_ok=True)
        f = open(os.path.join(f'{splits_dir}/data/splits', f'cifar100_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    if pseudo_lbl is not None:
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #balance the labeled and unlabeled data 
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    train_lbl_dataset = CIFAR100SSL(
        root,
        lbl_idx,
        train=True,
        transform=transform_train,
        pseudo_idx=pseudo_idx,
        pseudo_target=pseudo_target,
        nl_idx=nl_idx,
        nl_mask=nl_mask
    )
    
    if nl_idx is not None:
        train_nl_dataset = CIFAR100SSL(
            root,
            np.array(nl_idx),
            train=True,
            transform=transform_train,
            pseudo_idx=pseudo_idx,
            pseudo_target=pseudo_target,
            nl_idx=nl_idx,
            nl_mask=nl_mask
        )

    train_unlbl_dataset = CIFAR100SSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    val_dataset = CIFAR100SSL(root, indexs=val_idx, train=True, transform=transform_val, download=False)
    test_dataset = datasets.CIFAR100(root, train=False, transform=transform_val, download=False)

    if (nl_idx is not None) and (len(nl_idx) > 0):
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, val_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, val_dataset, test_dataset


def lbl_unlbl_split(idxs, lbls, n_lbl, n_class):
    lbl_per_class = n_lbl // n_class
    lbls = np.array(lbls)
    lbl_idx = []
    unlbl_idx = []
    for i in range(n_class):
        idx = idxs[np.where(lbls == i)[0]]
        np.random.shuffle(idx)
        lbl_idx.extend(idx[:lbl_per_class])
        unlbl_idx.extend(idx[lbl_per_class:])
    return lbl_idx, unlbl_idx


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.targets = np.array(self.targets)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
            
        if (nl_mask is not None) and (len(nl_mask) > 0):
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            indexs = np.array(indexs, dtype=np.int)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.nl_mask[index]


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.targets = np.array(self.targets)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
        
        if (nl_mask is not None) and (len(nl_mask) > 0):
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            indexs = np.array(indexs, dtype=np.int)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, self.indexs[index], self.nl_mask[index]