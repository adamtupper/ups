import argparse

import torch
from data.cifar import get_cifar10, get_cifar100
from torch.utils.data import DataLoader, SequentialSampler
from utils.evaluate import test
from utils.utils import *

parser = argparse.ArgumentParser(description='UPS Training')
parser.add_argument('--out', help='directory to output the result')
parser.add_argument('--data-dir', help='directory where the datasets are stored')
parser.add_argument('--exp-name', help='a unique ID for the experiment')
parser.add_argument('--no-restarts', action='store_true',
                    help="disable model resets between iterations")
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-gpu', default=1, type=int, help='number of gpus to use')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of workers')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'],
                    help='dataset names')
parser.add_argument('--n-lbl', type=int, default=4000,
                    help='number of labeled data')
parser.add_argument('--arch', default='cnn13', type=str,
                    choices=['wideresnet', 'cnn13', 'shakeshake'],
                    help='architecture name')
parser.add_argument('--iterations', default=20, type=int,
                    help='number of total pseudo-labeling iterations to run')
parser.add_argument('--epchs', default=1024, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
                    help='train batch_size')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    help='initial learning rate, default 0.03')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--wdecay', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=-1,
                    help="random seed (-1: don't use random seed)")
parser.add_argument('--no-progress', action='store_true',
                    help="don't use progress bar")
parser.add_argument('--dropout', default=0.3, type=float,
                    help='dropout probs')
parser.add_argument('--num-classes', default=10, type=int,
                    help='total classes')
parser.add_argument('--class-blnc', default=10, type=int,
                    help='total number of class balanced iterations')
parser.add_argument('--tau-p', default=0.70, type=float,
                    help='confidece threshold for positive pseudo-labels, default 0.70')
parser.add_argument('--tau-n', default=0.05, type=float,
                    help='confidece threshold for negative pseudo-labels, default 0.05')
parser.add_argument('--kappa-p', default=0.05, type=float,
                    help='uncertainty threshold for positive pseudo-labels, default 0.05')
parser.add_argument('--kappa-n', default=0.005, type=float,
                    help='uncertainty threshold for negative pseudo-labels, default 0.005')
parser.add_argument('--temp-nl', default=2.0, type=float,
                    help='temperature for generating negative pseduo-labels, default 2.0')
parser.add_argument('--no-uncertainty', action='store_true',
                    help='use uncertainty in the pesudo-label selection, default true')
parser.add_argument('--split-txt', default='run1', type=str,
                    help='extra text to differentiate different experiments. it also creates a new labeled/unlabeled split')
parser.add_argument('--model-width', default=2, type=int,
                    help='model width for WRN-28')
parser.add_argument('--model-depth', default=28, type=int,
                    help='model depth for WRN')
parser.add_argument('--test-freq', default=1, type=int,
                    help='frequency of evaluations')
parser.add_argument('--data-aug', type=str, required=True, help='Data augmentation setting (none, weak, or strong)')
parser.add_argument('--ckpt', type=str, required=True, help='The filename of the checkpoint to evaluate')


args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

DATASET_GETTERS = {'cifar10': get_cifar10, 'cifar100': get_cifar100}

_, _, _, val_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, args.out, args.data_dir, args.n_lbl, ssl_idx=None, seed=args.seed)

val_loader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=args.batch_size,
    num_workers=args.num_workers)

test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)

model = create_model(args)
model.to(args.device)

checkpoint = torch.load(f'{args.out}/{args.ckpt}')
model.load_state_dict(checkpoint['state_dict'])

val_loss, val_acc = test(args, val_loader, model)
print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

test_loss, test_acc = test(args, test_loader, model)
print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
