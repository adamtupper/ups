import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision import datasets, transforms

from .augmentations import CutoutRandom, RandAugment


def gcn(images, multiplier=55, eps=1e-10):
    """Performs global contrast normalization on a numpy array of images.
    
    From Oliver et al. (2018): github.com/brain-research/realistic-ssl-evaluation/
    
    Args:
        images: Numpy array of uint8s with shape = (num images, 3072). Each row of the
            array stores a 32x32 colour image. The first 1024 entries contain the red
            channel values, the next 1024 the green, and the final 1024 the blue. The
            image is stored in row-major order, so that the first 32 entries of the
            array are the red channel values of the first row of the image.
        multiplier: Post-normalization multiplier.
        eps: Small number for numerical stability.
        
    Returns:
        A numpy array of the same shape as images, but normalized.
    """
    images = images.astype(float)
    # Subtract the mean of image
    images -= images.mean(axis=1, keepdims=True)
    # Divide out the norm of each image
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    # Avoid divide-by-zero
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm


def get_zca_transformer(images, identity_scale=0.1, eps=1e-10):
    """Creates function performing ZCA normalization on a numpy array.
    
    From Oliver et al. (2018): github.com/brain-research/realistic-ssl-evaluation/
    
    Args:
        images: Numpy array of uint8s with shape = (num images, 3072). Each row of the
            array stores a 32x32 colour image. The first 1024 entries contain the red
            channel values, the next 1024 the green, and the final 1024 the blue. The
            image is stored in row-major order, so that the first 32 entries of the
            array are the red channel values of the first row of the image.
        identity_scale: Scalar multiplier for identity in SVD
        eps: Small constant to avoid divide-by-zero
        root_path: Optional path to save the ZCA params to.
    
    Returns:
        A function which applies ZCA to an array of flattened images
    """
    image_covariance = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(
        image_covariance + identity_scale * np.eye(*image_covariance.shape)
    )
    zca_decomp = np.dot(U, np.dot(np.diag(1. / np.sqrt(S + eps)), U.T))
    image_mean = images.mean(axis=0)

    return lambda x: np.dot(x - image_mean, zca_decomp)


def unflatten(images):
    """Takes a numpy array of flattened images with shape = (N, 3072) and reshapes them
    to (N, 32, 32, 3), where N is the number of images.

    Args:
        images (np.ndarray): Array with shape = (N, 3072). Each row of the
            array stores a 32x32 colour image. The first 1024 entries contain the red
            channel values, the next 1024 the green, and the final 1024 the blue. The
            image is stored in row-major order, so that the first 32 entries of the
            array are the red channel values of the first row of the image.

    Returns:
        np.ndarray: Array with shape = (N, 32, 32, 3) and RGB channel ordering.
    """
    return images.reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])


def flatten(images):
    """Takes a numpy array of images with shape = (N, 32, 32, 3) and reshapes them
    to (N, 3072), where N is the number of images.

    Args:
        images (np.ndarray): Array with shape = (N, 32, 32, 3) and RGB channel ordering.

    Returns:
        np.ndarray: Array with shape = (N, 3072). Each row of the array stores a 32 x 32
            colour image. The first 1024 entries contain the red channel values, the
            next 1024 the green, and the final 1024 the blue. The image is stored in
            row-major order, so that the first 32 entries of the array are the red
            channel values of the first row of the image.
    """
    return images.transpose([0, 3, 1, 2]).reshape((-1, 3072))


def get_cifar10(args, splits_dir=".", root='data/datasets', n_lbl=4000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt='', seed=None):
    os.makedirs(root, exist_ok=True)  # Create the root directory for saving data augmentations
    
    if args.zca is None and args.use_zca:
        images = datasets.CIFAR10(root, train=True, download=False).data
        images = images.reshape(-1, 3072)
        images = gcn(images)
        args.zca = get_zca_transformer(images)
    
    # Original augmentations
    # transform_train = transforms.Compose([
    #     RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
    #     transforms.ColorJitter(
    #         brightness=0.4,
    #         contrast=0.4,
    #         saturation=0.4,
    #     ),
    #     transforms.ToTensor(),
    #     CutoutRandom(n_holes=1, length=16, random=True),
    # ])
    
    # Augmentations from Oliver et al. (2018)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(2.0 / 32.0, 2.0 / 32.0)),  # 2 pixels
        transforms.Lambda(lambda x: x + torch.normal(mean=0.0, std=0.15, size=x.size())),  # Gaussian noise
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_idx = np.random.default_rng(seed=seed).choice(50000, size=45000, replace=False)
    val_idx = np.setdiff1d(np.arange(50000), train_idx)
    
    if ssl_idx is None:
        base_dataset = CIFAR10SSL(root, indexs=train_idx, zca_transform=args.zca, train=True, download=False)
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
        zca_transform=args.zca,
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
            zca_transform=args.zca,
            train=True,
            transform=transform_train,
            pseudo_idx=pseudo_idx,
            pseudo_target=pseudo_target,
            nl_idx=nl_idx,
            nl_mask=nl_mask
        )

    train_unlbl_dataset = CIFAR10SSL(
    root, train_unlbl_idx, zca_transform=args.zca, train=True, transform=transform_val)

    val_dataset = CIFAR10SSL(root, indexs=val_idx, zca_transform=args.zca, train=True, transform=transform_val, download=False)
    test_dataset = CIFAR10Preprocessed(root, zca_transform=args.zca, train=False, transform=transform_val, download=False)

    if (nl_idx is not None) and (len(nl_idx) > 0):
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, val_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, val_dataset, test_dataset


def get_cifar100(args, splits_dir=".", root='data/datasets', n_lbl=10000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt='', seed=None):
    ## augmentations
    transform_train = transforms.Compose([
        RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        CutoutRandom(n_holes=1, length=16, random=True)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    if ssl_idx is None:
        train_idx = np.random.default_rng(seed=seed).integers(50000, size=45000)
        val_idx = np.setdiff1d(np.arange(50000), train_idx)
        
        base_dataset = CIFAR100SSL(root, indexs=train_idx, train=True, download=False)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 100)
        
        os.makedirs(f'{splits_dir}/data/splits', exist_ok=True)
        f = open(os.path.join(f'{splits_dir}/data/splits', f'cifar100_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']
        val_idx = np.setdiff1d(np.arange(50000), np.concatenate((train_lbl_idx, train_unlbl_idx)))

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
        root, lbl_idx, train=True, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None:
        train_nl_dataset = CIFAR100SSL(
            root, np.array(nl_idx), train=True, transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = CIFAR100SSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    val_dataset = CIFAR10SSL(root, indexs=val_idx, zca_transform=args.zca, train=True, transform=transform_val)
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


class CIFAR10Preprocessed(datasets.CIFAR10):
    def __init__(self, root, zca_transform=None, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        # Flatten images
        images = flatten(self.data)
        
        # Apply global contrast normalization (GCN)
        self.data = gcn(images)
        
        if zca_transform is not None:
            # Apply ZCA whitening
            flattened_images = zca_transform(images)
        
        # Unflatten images
        self.data = unflatten(images)

class CIFAR10SSL(CIFAR10Preprocessed):
    def __init__(self, root, indexs, zca_transform, train=True,
                 transform=None, target_transform=None,
                 download=False, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, zca_transform=zca_transform, train=train,
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