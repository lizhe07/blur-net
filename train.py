import os
import time
import random
import json
import argparse
import torch
import torchvision

from jarvis import BaseJob
from jarvis.utils import (
    get_seed, set_seed, sgd_optimizer, cyclic_scheduler,
    time_str, progress_str, numpy_dict, tensor_dict,
)
from jarvis.vision import prepare_datasets, prepare_model, evaluate, MODELS

from blurnet.models import BlurNet

DEVICE = 'cuda'
NUM_WORKERS = 8
EVAL_BATCH_SIZE = 64
TRAIN_NUM_INFOS = 6
SAVE_INTERVAL = 8
NUM_EPOCHS = 48

TASKS = ['CIFAR10', 'CIFAR100', 'ImageNet']
ARCHS = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

class TrainingJob(BaseJob):
    r"""Base training job."""

    def __init__(self,
        store_dir: str = 'store',
        datasets_dir: str = 'datasets',
        *,
        device: str = DEVICE,
        num_workers: int = NUM_WORKERS,
        eval_batch_size: int = EVAL_BATCH_SIZE,
        train_num_infos: int = TRAIN_NUM_INFOS,
        save_interval: int = SAVE_INTERVAL,
        **kwargs,
    ):
        super(TrainingJob, self).__init__(store_dir=store_dir, **kwargs)
        self.datasets_dir = datasets_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_workers = num_workers
        self.eval_batch_size = eval_batch_size
        self.train_num_infos = train_num_infos
        self.save_interval = save_interval

    def strs2config(self, arg_strs=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', default='CIFAR10', choices=TASKS,
                            help="classification task")
        parser.add_argument('--arch', default='ResNet18', choices=ARCHS,
                            help="model architecture")
        parser.add_argument('--sigma', type=float,
                            help="standard deviation of Gaussian blurring kernel")
        parser.add_argument('--pretrain', action='store_true',
                            help="initialize from pretrained model")
        parser.add_argument('--seed', default=0, type=int,
                            help="random seed")
        parser.add_argument('--split-ratio', default=0.95, type=float,
                            help="ratio of training set split")
        parser.add_argument('--batch-size', default=32, type=int,
                            help="training batch size")
        parser.add_argument('--lr', default=0.01, type=float,
                            help="learning rate")
        parser.add_argument('--momentum', default=0.9, type=float,
                            help="momentum for SGD optimizer")
        parser.add_argument('--weight-decay', default=1e-4, type=float,
                            help="weight decay")
        parser.add_argument('--phase-len', default=4, type=int,
                            help="length of learning phases")
        parser.add_argument('--num-phases', default=3, type=int,
                            help="number of phases in each cycle")
        parser.add_argument('--gamma', default=0.3, type=float,
                            help="learning rate decay between phases")
        args, _ = parser.parse_known_args(arg_strs)

        model_config = {
            'task': args.task,
            'arch': args.arch,
            'sigma': args.sigma,
        }
        train_config = {
            'pretrain': args.pretrain,
            'seed': get_seed(args.seed),
            'split_ratio': args.split_ratio,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'phase_len': args.phase_len,
            'num_phases': args.num_phases,
            'gamma': args.gamma,
        }
        return {
            'model_config': model_config,
            'train_config': train_config,
        }

    def prepare_model(self, model_config):
        def blurnet(**kwargs):
            resnet = MODELS[model_config['arch']](**kwargs)
            return BlurNet(resnet, model_config['sigma'])
        model = prepare_model(
            task=model_config['task'], arch=blurnet,
            conv0_kernel_size=3 if model_config['task'].startswith('CIFAR') else 7,
        )
        return model

    def train(self, model, optimizer, dataset, batch_size, verbose=1):
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

        Returns
        -------
        running_loss, running_acc: float
            The running average of loss and accuracy of training batches.

        """
        model.train().to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers,
        )

        num_batches = len(loader)
        running_loss, running_acc = None, None
        for b_idx, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(self.device), labels.to(self.device)
            logits = model(images)
            loss = criterion(logits, labels)
            _, predicts = logits.max(dim=1)
            acc = (predicts==labels).to(torch.float).mean()

            running_loss = loss.item() if running_loss is None else 0.99*running_loss+0.01*loss.item()
            running_acc = acc.item() if running_acc is None else 0.99*running_acc+0.01*acc.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose>0 and (b_idx%(-(-num_batches//self.train_num_infos))==0 or b_idx==num_batches):
                print('{}: [loss: {:5.3f}] [acc:{:7.2%}]'.format(
                    progress_str(b_idx, num_batches, True), running_loss, running_acc,
                ))
        return running_loss, running_acc

    def evaluate(self, model, dataset, verbose):
        loss, acc = evaluate(
            model, dataset, batch_size=self.eval_batch_size, device=self.device, verbose=verbose,
        )
        return loss, acc

    def _download_resnet(self, arch):
        if arch=='ResNet18':
            resnet = torchvision.models.resnet18(pretrained=True, progress=False)
        if arch=='ResNet34':
            resnet = torchvision.models.resnet34(pretrained=True, progress=False)
        if arch=='ResNet50':
            resnet = torchvision.models.resnet50(pretrained=True, progress=False)
        if arch=='ResNet101':
            resnet = torchvision.models.resnet101(pretrained=True, progress=False)
        if arch=='ResNet152':
            resnet = torchvision.models.resnet152(pretrained=True, progress=False)
        return resnet

    def main(self, config, num_epochs=1, verbose=1):
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

        # prepare model, optimizer and scheduler
        model = self.prepare_model(model_config)
        if train_config['pretrain'] and model_config['task']=='ImageNet':
            resnet = self._download_resnet(model_config['arch'])
            model.resnet.load_pytorch_model(resnet.state_dict())
        model.to(self.device)
        optimizer = sgd_optimizer(
            model, train_config['lr'], train_config['momentum'], train_config['weight_decay'],
        )
        scheduler = cyclic_scheduler(
            optimizer, train_config['phase_len'], train_config['num_phases'], train_config['gamma'],
        )

        # train until completed
        try:
            epoch, ckpt = self.load_ckpt(config)
            losses, accs = ckpt['losses'], ckpt['accs']
            min_loss, best_epoch, best_state = ckpt['min_loss'], ckpt['best_epoch'], ckpt['best_state']
            model.load_state_dict(tensor_dict(ckpt['model_state'], self.device))
            optimizer.load_state_dict(tensor_dict(ckpt['optimizer_state'], self.device))
            scheduler.load_state_dict(tensor_dict(ckpt['scheduler_state'], self.device))
            if verbose>0:
                print(f"Checkpoint (epoch {epoch}) loaded successfully.")
        except:
            epoch = 0
            losses = {'train': [], 'valid': [], 'test': []}
            accs = {'train': [], 'valid': [], 'test': []}
            min_loss, best_epoch, best_state = float('inf'), 0, None
            if verbose>0:
                print("No checkpoint loaded.")
        if torch.cuda.device_count()>1:
            if verbose>0:
                print("Using {} GPUs".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
            use_dp = True
        else:
            use_dp = False
        while epoch<num_epochs:
            if verbose>0:
                print(f"\nEpoch {epoch}")
                print("Training...")
                print("lr: {:.2g}".format(optimizer.param_groups[0]['lr']))
            tic = time.time()
            loss, acc = self.train(
                model, optimizer, dataset_train, train_config['batch_size'], verbose,
            )
            scheduler.step()
            losses['train'].append(loss)
            accs['train'].append(acc)
            toc = time.time()
            if verbose>0:
                print("loss: {:5.3f}, acc:{:7.2%} ({})".format(loss, acc, time_str(toc-tic)))

            if verbose>0:
                print("Evaluating...")
            for key, dataset in zip(['valid', 'test'], [dataset_valid, dataset_test]):
                if verbose>0:
                    if key=='valid':
                        print("Validation set:")
                    if key=='test':
                        print("Testing set:")
                loss, acc = self.evaluate(model, dataset, verbose)
                losses[key].append(loss)
                accs[key].append(acc)
            if losses['valid'][epoch]<min_loss:
                min_loss = losses['valid'][epoch]
                best_epoch = epoch
                best_state = numpy_dict(model.state_dict() if not use_dp else model.module.state_dict())
            epoch += 1

            ckpt = {
                'losses': losses, 'accs': accs,
                'min_loss': min_loss, 'best_epoch': best_epoch, 'best_state': best_state,
                'model_state': numpy_dict(model.state_dict() if not use_dp else model.module.state_dict()),
                'optimizer_state': numpy_dict(optimizer.state_dict()),
                'scheduler_state': numpy_dict(scheduler.state_dict()),
            }
            preview = {
                'loss_train': losses['train'][best_epoch],
                'loss_valid': losses['valid'][best_epoch],
                'loss_test': losses['test'][best_epoch],
                'acc_train': accs['train'][best_epoch],
                'acc_valid': accs['valid'][best_epoch],
                'acc_test': accs['test'][best_epoch],
            }
            if epoch%self.save_interval==0:
                self.save_ckpt(config, epoch, ckpt, preview)
        if verbose>0:
            print("\nTesting accuracy at best epoch {:.2%}".format(accs['test'][best_epoch]))
        return ckpt, preview

    def export(self, key, export_path=None):
        r"""Exports a trained model."""
        if export_path is None:
            os.makedirs(f'{self.store_dir}/exported', exist_ok=True)
            export_path = f'{self.store_dir}/exported/{key}.pt'
        config = self.configs[key].native()
        model = self.prepare_model(config['model_config'])
        model.load_state_dict(tensor_dict(self.ckpts[key]['best_state']))
        preview = self.previews[key]

        torch.save({
            'config': config,
            'task': config['model_config']['task'],
            'model': model,
            'acc': preview['acc_test'],
        }, export_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='store')
    parser.add_argument('--datasets-dir', default='datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--eval-batch-size', default=EVAL_BATCH_SIZE, type=int)
    parser.add_argument('--train-num-infos', default=TRAIN_NUM_INFOS, type=int)
    parser.add_argument('--save-interval', default=SAVE_INTERVAL, type=int)
    parser.add_argument('--spec-path', default='jsons/C10R18_train.json')
    parser.add_argument('--max-wait', default=1, type=float, help="seconds of wait before each job")
    parser.add_argument('--num-works', default=0, type=int, help="number of works to process")
    parser.add_argument('--patience', default=168, type=float, help="hours since last modification")
    parser.add_argument('--num-epochs', default=NUM_EPOCHS, type=int)
    args, arg_strs = parser.parse_known_args()

    job = TrainingJob(
        args.store_dir, args.datasets_dir,
        device=args.device,
        eval_batch_size=args.eval_batch_size,
        train_num_infos=args.train_num_infos,
        save_interval=args.save_interval,
    )

    if args.spec_path is None:
        args.spec_path = 'jsons/C10R18_train.json'
    with open(args.spec_path, 'r') as f:
        search_spec = json.load(f)
    time.sleep(random.random()*args.max_wait)
    job.grid_search(
        search_spec, num_epochs=args.num_epochs, num_works=args.num_works, patience=args.patience,
    )
