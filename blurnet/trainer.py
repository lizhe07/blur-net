# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 00:22:02 2020

@author: Zhe
"""

import argparse, time
import torch
from torch.utils.data import DataLoader

from jarvis import BaseJob
from jarvis.vision import prepare_datasets, evaluate
from jarvis.utils import get_seed, set_seed, update_default, \
    numpy_dict, tensor_dict, progress_str, time_str

from . import __version__ as VERSION
from . import models

DEVICE = 'cuda'
EVAL_BATCH_SIZE = 160
TRAIN_DISP_NUM = 6
WORKER_NUM = 0

MODELS = {
    'ResNet18': models.blurnet18,
    'ResNet34': models.blurnet34,
    'ResNet50': models.blurnet50,
    'ResNet101': models.blurnet101,
    'ResNet152': models.blurnet152,
    }


class BlurJob(BaseJob):

    def __init__(
            self, store_dir=None, datasets_dir='vision_datasets',
            device=DEVICE, eval_batch_size=EVAL_BATCH_SIZE,
            train_disp_num=TRAIN_DISP_NUM, worker_num=WORKER_NUM,
            ):
        super(BlurJob, self).__init__(store_dir)
        self.datasets_dir = datasets_dir
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.train_disp_num = train_disp_num
        self.worker_num = worker_num

    def get_config(self, arg_strs):
        return get_config(arg_strs)

    def main(self, config, verbose=True):
        model_config = config['model_config']
        train_config = config['train_config']

        print('model config:\n{}'.format(model_config))
        print('train config:\n{}'.format(train_config))
        set_seed(train_config['seed'])

        # prepare task datasets
        dataset_train, dataset_valid, dataset_test, weight = prepare_datasets(
            model_config['task'], self.datasets_dir, train_config['valid_num'],
            grayscale=model_config['grayscale'],
            )
        # prepare model
        model = prepare_model(**model_config)
        if model_config['sigma'] is None:
            print('\n{} model for {} initialized'.format(
                model_config['arch'], model_config['task'],
                ))
        else:
            print('\n{} model for {} initialized (blur {:g})'.format(
                model_config['arch'], model_config['task'], model_config['sigma'],
                ))

        # prepare optimizer and scheduler
        params = []
        params.append({
            'params': [param for name, param in model.named_parameters() if name.endswith('weight')],
            'weight_decay': train_config['weight_decay'],
            })
        params.append({
            'params': [param for name, param in model.named_parameters() if not name.endswith('weight')],
            })
        optimizer = torch.optim.SGD(
            params, lr=train_config['lr'],
            momentum=train_config['momentum'],
            )

        # train until completed
        epoch_idx, epoch_num = 0, train_config['epoch_num']
        losses = {'valid': [], 'test': []}
        accs = {'valid': [], 'test': []}
        states = []
        while True:
            tic = time.time()
            # evaluate task performance on validation and testing set
            print('evaluating task performance...')
            for key, dataset in zip(['valid', 'test'], [dataset_valid, dataset_test]):
                if key=='valid':
                    print('validation set:')
                if key=='test':
                    print('testing set:')
                loss, acc = evaluate(
                    model, dataset, device=self.device,
                    batch_size=self.eval_batch_size,
                    worker_num=self.worker_num,
                    )
                print('loss: {:4.2f}, acc:{:7.2%}'.format(loss, acc))
                losses[key].append(loss)
                accs[key].append(acc)
            # save model parameters
            states.append(numpy_dict(model.state_dict()))
            best_epoch = losses['valid'].index(min(losses['valid']))
            toc = time.time()
            print('elapsed time for evaluation: {}'.format(time_str(toc-tic)))

            # adjust learning rate and reload from checkpoints
            if epoch_idx in [int(0.5*epoch_num), int(0.8*epoch_num)]:
                for p_group in optimizer.param_groups:
                    p_group['lr'] *= 0.1
                model.load_state_dict(tensor_dict(states[best_epoch]))
                print('learning rate decreased, and best model so far reloaded')

            epoch_idx += 1
            if epoch_idx>epoch_num:
                break
            print('\nepoch {}'.format(epoch_idx))

            tic = time.time()
            print('training...')
            print('lr: {:.4g}'.format(optimizer.param_groups[0]['lr']))
            train(
                model, optimizer, dataset_train, weight, train_config['batch_size'],
                self.device, self.train_disp_num, self.worker_num,
                )
            toc = time.time()
            print('elapsed time for one epoch: {}'.format(time_str(toc-tic)))

        print('\ntest acc at best epoch ({}) {:.2%}'.format(best_epoch, accs['test'][best_epoch]))

        output = {
            'losses': losses,
            'accs': accs,
            'best_state': states[best_epoch],
            'best_epoch': best_epoch,
            }
        preview = {
            'best_epoch': best_epoch,
            'loss_valid': losses['valid'][best_epoch],
            'loss_test': losses['test'][best_epoch],
            'acc_valid': accs['valid'][best_epoch],
            'acc_test': accs['test'][best_epoch],
            }
        return output, preview


def prepare_model(task, grayscale, arch, sigma):
    r"""Prepares model.

    Args
    ----
    task: str
        The name of the dataset, e.g. ``'CIFAR10'``.
    arch: str
        The name of model architecture, e.g. ``ResNet18``.
    sigma: float
        The standard deviaiont of Gaussian blurring kernel.

    """
    if task=='CIFAR10':
        class_num = 10
    if task=='CIFAR100':
        class_num = 100
    if task=='16ImageNet':
        class_num = 16
    if task=='ImageNet':
        class_num = 1000
    in_channels = 1 if grayscale else 3

    model = MODELS[arch](in_channels=in_channels, class_num=class_num, sigma=sigma)
    return model


def train(model, optimizer, dataset, weight, batch_size, device,
          disp_num=TRAIN_DISP_NUM, worker_num=WORKER_NUM):
    r"""Trains the model for one epoch.

    Args
    ----
    model: nn.Module
        The model to be trained.
    optimizer: Optimizer
        The optimizer for `model`.
    dataset: Dataset
        The classification dataset.
    weight: (class_num,), tensor
        The class weight for unbalanced training set.
    batch_size: int
        The batch size of training.
    device: int
        The device for training.
    disp_num: int
        The display number for one training epoch.
    worker_num: int
        The number of workers of the data loader.

    """
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.train().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight).to(device)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=worker_num
        )

    batch_num = len(loader)
    for batch_idx, (images, labels) in enumerate(loader, 1):
        logits = model(images.to(device))
        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%(-(-batch_num//disp_num))==0 or batch_idx==batch_num:
            with torch.no_grad():
                loss = criterion(logits, labels.to(device))
                _, predicts = logits.max(dim=1)
                acc = (predicts.cpu()==labels).to(torch.float).mean()
            print('{}: [loss: {:4.2f}] [acc:{:7.2%}]'.format(
                progress_str(batch_idx, batch_num, True),
                loss.item(), acc.item(),
                ))


def get_config(arg_strs=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', '16ImageNet', 'ImageNet'])
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--arch', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'])
    parser.add_argument('--sigma', type=float)

    parser.add_argument('--train_seed', type=int)
    parser.add_argument('--valid_num', type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epoch_num', default=50, type=int)

    args, arg_strs = parser.parse_known_args(arg_strs)

    model_config = {
        'task': args.task,
        'grayscale': args.grayscale,
        'arch': args.arch,
        'sigma': args.sigma,
        }

    if args.valid_num is None:
        if model_config['task'].startswith('CIFAR'):
            args.valid_num = 5000
        if model_config['task']=='16ImageNet':
            args.valid_num = 100
        if model_config['task']=='ImageNet':
            args.valid_num = 50
    train_config = {
        'seed': get_seed(args.train_seed),
        'valid_num': args.valid_num,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'epoch_num': args.epoch_num,
        }

    return {
        'model_config': model_config,
        'train_config': train_config,
        }
