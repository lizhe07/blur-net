# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 00:22:02 2020

@author: Zhe
"""

import os, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jarvis import BaseJob
from jarvis.vision import prepare_datasets, prepare_model, evaluate
from jarvis.utils import get_seed, set_seed, update_default, \
    numpy_dict, tensor_dict, progress_str, time_str

from . import __version__ as VERSION

EVAL_DEVICE = 'cuda'
EVAL_BATCH_SIZE = 64
TRAIN_DISP_NUM = 6
WORKER_NUM = 4


class BlurJob(BaseJob):

    def __init__(self, save_dir, benchmarks_dir):
        super(BlurJob, self).__init__(save_dir)
        self.benchmarks_dir = benchmarks_dir

    def get_work_config(self, arg_strs):
        model_config, train_config = get_configs(arg_strs)

        work_config = {
            'model_config': model_config,
            'train_config': train_config,
            }
        return work_config

    def main(self, work_config):
        run_config = {
            'benchmarks_dir': self.benchmarks_dir,
            }

        losses, accs, states, best_epoch = main(
            **work_config, run_config=run_config
            )

        output = {
            'losses': losses,
            'accs': accs,
            'best_state': states[best_epoch],
            'best_epoch': best_epoch,
            }
        preview = {
            'loss_valid': losses['valid'][best_epoch],
            'loss_test': losses['test'][best_epoch],
            'acc_valid': accs['valid'][best_epoch],
            'acc_test': accs['test'][best_epoch],
            }
        return output, preview

    def export_best(self, model_config, top_k=5):
        matched_ids, losses = [], []
        for w_id in self.completed_ids():
            config = self.configs.fetch_record(w_id)
            if config['model_config']==model_config:
                matched_ids.append(w_id)
                losses.append(self.previews.fetch_record(w_id)['loss_test'])
        best_ids = [w_id for i, (w_id, _) in enumerate(sorted(
            zip(matched_ids, losses), key=lambda item:item[1], reverse=True)) if i<top_k]

        export_dir = os.path.join(self.save_dir, 'exported')
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        for w_id in best_ids:
            config = self.configs.fetch_record(w_id)
            output = self.outputs.fetch_record(w_id)
            model = prepare_blur_model(**config['model_config'])
            model.load_state_dict(tensor_dict(output['best_state']))

            saved = {
                'version': VERSION,
                'config': config,
                'model': model,
                'losses': output['losses'],
                'accs': output['accs'],
                'best_epoch': output['best_epoch'],
                }
            torch.save(saved, os.path.join(export_dir, '{}.pt'.format(w_id)))
        return matched_ids


def prepare_blur_model(task, arch, blur_sigma):
    base_model = prepare_model(task, arch)

    if blur_sigma is None:
        return base_model
    else:
        half_size = int(-(-2.5*blur_sigma//1))
        x, y = torch.meshgrid(
            torch.arange(-half_size, half_size+1).to(torch.float),
            torch.arange(-half_size, half_size+1).to(torch.float)
            )
        w = torch.exp(-(x**2+y**2)/(2*blur_sigma**2))
        w /= w.sum()
        blur = nn.Conv2d(3, 3, 2*half_size+1, padding=half_size,
                         padding_mode='zeros', bias=False)
        blur.weight.data *= 0
        for i in range(3):
            blur.weight.data[i, i] = w
        blur.weight.requires_grad = False
        blur_model = nn.Sequential(
            blur, base_model
            )
        return blur_model


def train(model, optimizer, dataset, weight, batch_size, device,
          disp_num=TRAIN_DISP_NUM, worker_num=WORKER_NUM):
    r"""Trains the model for one epoch.

    Args
    ----
    model: nn.Module
        The model to be trained.
    optimizer: Optimizer
        The optimizer for `model`.
    task_dataset: Dataset
        The classification task dataset.
    weight: (class_num,), tensor
        The class weight for unbalanced training set.
    train_config: dict
        The training configuration dictionary.
    reg_config: dict
        The regularization configuration dictionary.
    beta: float
        The damping coefficient for updating mean activation.
    eps: float
        The small positive number used in similarity loss.
    disp_num: int
        The display number for one training epoch.
    worker_num: int
        The number of workers of the data loader.

    """
    model.train().to(device)
    criterion_task = torch.nn.CrossEntropyLoss(weight=weight).to(device)
    loader_task = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=worker_num
        )

    batch_num = len(loader_task)
    for batch_idx, (task_images, task_labels) in enumerate(loader_task, 1):
        logits = model(task_images.to(device))
        loss = criterion_task(logits, task_labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%(-(-batch_num//disp_num))==0 or batch_idx==batch_num:
            with torch.no_grad():
                _, predicts = logits.max(dim=1)
                flags = (predicts.cpu()==task_labels).to(torch.float)
                if weight is None:
                    acc = flags.mean()
                else:
                    acc = (flags*weight[task_labels]).sum()/weight[task_labels].sum()
            print('{}: [loss: {:4.2f}] [acc:{:7.2%}]'.format(
                progress_str(batch_idx, batch_num, True),
                loss.item(), acc.item(),
                ))


def get_configs(arg_strs=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', '16ImageNet', 'ImageNet'])
    parser.add_argument('--arch', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'])
    parser.add_argument('--blur_sigma', type=float)

    parser.add_argument('--train_device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--train_seed', type=int)
    parser.add_argument('--valid_num', type=float)
    parser.add_argument('--task_batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epoch_num', default=50, type=int)

    args, arg_strs = parser.parse_known_args(arg_strs)

    model_config = {
        'task': args.task,
        'arch': args.arch,
        'blur_sigma': args.blur_sigma,
        }

    if args.valid_num is None:
        if model_config['task'].startswith('CIFAR'):
            args.valid_num = 5000
        if model_config['task']=='16ImageNet':
            args.valid_num = 100
        if model_config['task']=='ImageNet':
            args.valid_num = 50000
    train_config = {
        'device': args.train_device,
        'seed': get_seed(args.train_seed),
        'valid_num': args.valid_num,
        'batch_size': args.task_batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'epoch_num': args.epoch_num,
        }

    return model_config, train_config


def main(model_config, train_config, run_config=None):
    print('model config:\n{}'.format(model_config))
    print('train config:\n{}'.format(train_config))
    run_config = update_default({
        'benchmarks_dir': 'benchmarks',
        'eval_device': EVAL_DEVICE,
        'eval_batch_size': EVAL_BATCH_SIZE,
        'train_disp_num': TRAIN_DISP_NUM,
        'worker_num': WORKER_NUM,
        }, run_config)
    set_seed(train_config['seed'])

    # prepare task datasets
    dataset_train, dataset_valid, dataset_test, weight = \
        prepare_datasets(model_config['task'], run_config['benchmarks_dir'],
                         train_config['valid_num'])
    # prepare model
    model = prepare_blur_model(model_config['task'], model_config['arch'],
                               model_config['blur_sigma'])
    print('\n{} model for {} initialized (blur {:g})'.format(
        model_config['arch'], model_config['task'], model_config['blur_sigma'],
        ))

    # prepare optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=train_config['lr'],
        momentum=train_config['momentum'],
        weight_decay=train_config['weight_decay']
        )

    # train until completed
    epoch_idx = 0
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
                model, dataset,
                device=run_config['eval_device'],
                batch_size=run_config['eval_batch_size'],
                worker_num=run_config['worker_num'],
                )
            print('loss: {:4.2f}, acc:{:7.2%}'.format(loss, acc))
            losses[key].append(loss)
            accs[key].append(acc)
        # save model parameters
        states.append(numpy_dict(model.state_dict()))
        best_epoch = losses['valid'].index(min(losses['valid']))
        toc = time.time()
        print('elapsed time for evaluation: {}'.format(time_str(toc-tic)))

        epoch_idx += 1
        if epoch_idx>train_config['epoch_num']:
            break
        print('\nepoch {}'.format(epoch_idx))

        # adjust learning rate and reload from checkpoints
        if epoch_idx in [int(0.5*train_config['epoch_num'])+1,
                         int(0.8*train_config['epoch_num'])+1,
                         train_config['epoch_num']]:
            optimizer.param_groups[0]['lr'] *= 0.1
            model.load_state_dict(tensor_dict(states[best_epoch]))
            print('learning rate decreased, and best model so far reloaded')

        tic = time.time()
        print('training...')
        print('lr: {:.4g}'.format(optimizer.param_groups[0]['lr']))
        train(
            model, optimizer, dataset_train, weight,
            train_config['batch_size'], train_config['device'],
            disp_num=run_config['train_disp_num'],
            worker_num=run_config['worker_num']
            )
        toc = time.time()
        print('elapsed time for one epoch: {}'.format(time_str(toc-tic)))

    print('\ntest acc at best epoch ({}) {:.2%}'.format(best_epoch, accs['test'][best_epoch]))
    return losses, accs, states, best_epoch
