# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 00:22:02 2020

@author: Zhe
"""

import os, argparse, time, pickle, torch

from jarvis import BaseJob
from jarvis.vision import prepare_datasets, prepare_model, evaluate
from jarvis.utils import (
    get_seed, set_seed, job_parser, sgd_optimizer, cyclic_scheduler,
    time_str, progress_str, numpy_dict, tensor_dict,
    )

from . import __version__ as VERSION
from . import models

DEVICE = 'cuda'
WORKER_NUM = 0
EVAL_BATCH_SIZE = 64
TRAIN_DISP_NUM = 6

TASKS = ['CIFAR10', 'CIFAR100', 'ImageNet']
ARCHS = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
MODELS = {
    'ResNet18': models.blurnet18,
    'ResNet34': models.blurnet34,
    'ResNet50': models.blurnet50,
    'ResNet101': models.blurnet101,
    'ResNet152': models.blurnet152,
    }


class TrainJob(BaseJob):

    def __init__(
            self, store_dir, datasets_dir, *, device=DEVICE,
            worker_num=WORKER_NUM, eval_batch_size=EVAL_BATCH_SIZE,
            train_disp_num=TRAIN_DISP_NUM,
            ):
        if store_dir is None:
            super(TrainJob, self).__init__()
        else:
            super(TrainJob, self).__init__(os.path.join(store_dir, 'models'))
        self.datasets_dir = datasets_dir

        self.device = 'cuda' if device=='cuda' and torch.cuda.is_available() else 'cpu'
        self.worker_num = worker_num
        self.eval_batch_size = eval_batch_size
        self.train_disp_num = train_disp_num

    def prepare_model(self, model_config):
        model = prepare_model(
            model_config['task'], model_config['arch'],
            sigma=model_config['sigma'],
            conv0_kernel_size=3 if model_config['task'].startswith('CIFAR') else 7,
            )
        return model

    def get_config(self, arg_strs=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', default='CIFAR10', choices=TASKS,
                            help="classification task")
        parser.add_argument('--arch', default='ResNet18', choices=ARCHS,
                            help="model architecture")
        parser.add_argument('--sigma', type=float,
                            help="standard deviation of Gaussian blurring kernel")
        parser.add_argument('--seed', default=0, type=int,
                            help="random seed")
        parser.add_argument('--split_ratio', default=0.9, type=float,
                            help="ratio of training set split")
        parser.add_argument('--batch_size', default=32, type=int,
                            help="training batch size")
        parser.add_argument('--lr', default=0.01, type=float,
                            help="learning rate")
        parser.add_argument('--momentum', default=0.9, type=float,
                            help="momentum for SGD optimizer")
        parser.add_argument('--weight_decay', default=5e-4, type=float,
                            help="weight decay")
        parser.add_argument('--epoch_num', default=24, type=int,
                            help="number of epochs")
        parser.add_argument('--cycle_num', default=2, type=int,
                            help="number of learning rate cycles")
        parser.add_argument('--phase_num', default=3, type=int,
                            help="number of phases in each learning rate cycle")
        parser.add_argument('--gamma', default=0.3, type=float,
                            help="learning rate decay between phases")
        args, _ = parser.parse_known_args(arg_strs)

        model_config = {
            'task': args.task,
            'arch': args.arch,
            'sigma': args.sigma,
            }
        train_config = {
            'seed': get_seed(args.seed),
            'split_ratio': args.split_ratio,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'epoch_num': args.epoch_num,
            'cycle_num': args.cycle_num,
            'phase_num': args.phase_num,
            'gamma': args.gamma,
            }
        return {
            'model_config': model_config,
            'train_config': train_config,
            }

    def main(self, config, verbose=True):
        model_config = config['model_config']
        train_config = config['train_config']
        if verbose:
            print(f"model config:\n{model_config}")
            print(f"train config:\n{train_config}")
        set_seed(train_config['seed'])

        # prepare task datasets
        dataset_train, dataset_valid, dataset_test = prepare_datasets(
            model_config['task'], self.datasets_dir, train_config['split_ratio'],
            )

        # prepare model
        model = self.prepare_model(model_config)

        # prepare optimizer and scheduler
        optimizer = sgd_optimizer(
            model, train_config['lr'],
            train_config['momentum'], train_config['weight_decay'],
            )
        scheduler = cyclic_scheduler(
            optimizer, train_config['epoch_num'], train_config['cycle_num'],
            train_config['phase_num'], train_config['gamma'],
            )

        # train until completed
        epoch_idx, epoch_num = 0, train_config['epoch_num']
        losses = {'train': [], 'valid': [], 'test': []}
        accs = {'train': [], 'valid': [], 'test': []}
        states = []
        while True:
            # evaluate task performance on validation and testing set
            if verbose:
                tic = time.time()
                print("evaluating performance...")
            for key, dataset in zip(['valid', 'test'], [dataset_valid, dataset_test]):
                if key=='valid':
                    print("validation set:")
                if key=='test':
                    print("testing set:")
                loss, acc = evaluate(
                    model, dataset, batch_size=self.eval_batch_size,
                    device=self.device, worker_num=self.worker_num,
                    )
                if verbose:
                    print("loss: {:5.3f}, acc:{:7.2%}".format(loss, acc))
                losses[key].append(loss)
                accs[key].append(acc)
            if verbose:
                toc = time.time()
                print("elapsed time: {}".format(time_str(toc-tic)))
            # save model parameters
            states.append(numpy_dict(model.state_dict()))

            epoch_idx += 1
            if epoch_idx>epoch_num:
                break

            # train one epoch
            if verbose:
                print("\nepoch {}\ntraining...".format(epoch_idx))
                print("lr: {:.2g}".format(optimizer.param_groups[0]['lr']))
                tic = time.time()
            loss, acc = self.train(
                model, optimizer, dataset_train, train_config['batch_size'], verbose,
                )
            losses['train'].append(loss)
            accs['train'].append(acc)
            if verbose:
                toc = time.time()
                print("loss: {:5.3f}, acc:{:7.2%}".format(loss, acc))
                print("elapsed time for one epoch: {}".format(time_str(toc-tic)))
            scheduler.step()

        best_epoch = losses['valid'].index(min(losses['valid']))
        if verbose:
            print("\ntest acc at best epoch ({}) {:.2%}".format(best_epoch, accs['test'][best_epoch]))
        result = {
            'losses': losses,
            'accs': accs,
            'last_state': states[-1],
            'best_state': states[best_epoch],
            'best_epoch': best_epoch,
            }
        preview = {
            'loss_train': losses['train'][best_epoch-1],
            'loss_valid': losses['valid'][best_epoch],
            'loss_test': losses['test'][best_epoch],
            'acc_train': accs['train'][best_epoch-1],
            'acc_valid': accs['valid'][best_epoch],
            'acc_test': accs['test'][best_epoch],
            }
        return result, preview

    def train(self, model, optimizer, dataset, batch_size, verbose=True):
        r"""Trains for one epoch.

        Args
        ----
        model: nn.Module
            The model to be trained.
        optimizer: Optimizer
            The optimizer for `model`.
        dataset: Dataset
            The training dataset.
        batch_size: int
            The batch size for training.
        verbose: bool
            Whether to display information.

        Returns
        -------
        running_loss, running_acc: float
            The running average of loss and accuracy of training batches.

        """
        device = self.device
        model.train().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=self.worker_num,
            )

        batch_num = len(loader)
        running_loss, running_acc = None, None
        for batch_idx, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            _, predicts = logits.max(dim=1)
            acc = (predicts==labels).to(torch.float).mean()

            running_loss = loss.item() if running_loss is None else 0.99*running_loss+0.01*loss.item()
            running_acc = acc.item() if running_acc is None else 0.99*running_acc+0.01*acc.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (batch_idx%(-(-batch_num//self.train_disp_num))==0 or batch_idx==batch_num):
                print('{}: [loss: {:4.2f}] [acc:{:7.2%}]'.format(
                    progress_str(batch_idx, batch_num, True),
                    loss.item(), acc.item(),
                    ))
        return running_loss, running_acc

    def export(self, key, export_pth):
        r"""Exports a trained model.

        Args
        ----
        key: str
            The key of the model.
        export_pth: str
            The path of exported file.

        """
        config = self.configs[key].native()
        preview = self.previews[key]
        model = self.prepare_model(config['model_config'])
        model.load_state_dict(tensor_dict(self.results[key]['best_state']))

        torch.save({
            'version': VERSION,
            'config': config,
            'task': config['model_config']['task'],
            'model': model,
            'acc': preview['acc_test'],
            }, export_pth)


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store_dir', default='store')
    parser.add_argument('--datasets_dir', default='datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
    parser.add_argument('--eval_batch_size', default=EVAL_BATCH_SIZE, type=int)
    parser.add_argument('--train_disp_num', default=TRAIN_DISP_NUM, type=int)
    args, arg_strs = parser.parse_known_args()

    job = TrainJob(
        args.store_dir, args.datasets_dir,
        device=args.device, worker_num=args.worker_num,
        eval_batch_size=args.eval_batch_size,
        train_disp_num=args.train_disp_num,
        )

    if args.spec_pth is None:
        job.process(job.get_config(arg_strs))
    else:
        with open(args.spec_pth, 'rb') as f:
            search_spec = pickle.load(f)
        job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
