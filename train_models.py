# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 02:35:48 2020

@author: Zhe
"""

import argparse, pickle, time, random
from blurnet import trainer

parser = argparse.ArgumentParser()

parser.add_argument('--store_dir', default='store')
parser.add_argument('--datasets_dir', default='vision_datasets')
parser.add_argument('--device', default=trainer.DEVICE)
parser.add_argument('--eval_batch_size', default=trainer.EVAL_BATCH_SIZE, type=int)
parser.add_argument('--train_disp_num', default=trainer.TRAIN_DISP_NUM, type=int)
parser.add_argument('--worker_num', default=trainer.WORKER_NUM, type=int)

parser.add_argument('--spec_pth')
parser.add_argument('--max_wait', default=1, type=float)
parser.add_argument('--process_num', default=0, type=int)
parser.add_argument('--tolerance', default=float('inf'), type=float)

args = parser.parse_args()

if __name__=='__main__':
    job = trainer.BlurJob(
        args.store_dir, datasets_dir=args.datasets_dir,
        device=args.device, eval_batch_size=args.eval_batch_size,
        train_disp_num=args.train_disp_num, worker_num=args.worker_num,
        )

    with open(args.spec_pth, 'rb') as f:
        search_spec = pickle.load(f)
    random_wait = random.random()*args.max_wait
    print('random wait {:.1f}s'.format(random_wait))
    time.sleep(random_wait)

    job.random_search(search_spec, args.process_num, args.tolerance)
