import os
import argparse
import random
import numpy as np
from HyerIQASolver import HyperIQASolver
import scipy.io as scio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(config):
    folder_path = {
        'live': '/home/ssl/Database/databaserelease2/',
        'csiq': '/home/ssl/Database/CSIQ/',
        'tid': 'D:\ImageDatabase\\tid2013',
        'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        'koniq-10k': 'D:\ImageDatabase\koniq10k_1024x768',
        'bid': '/home/ssl/Database/BID/',
        'kadid': 'H:\kadis700k\kadis700k',
        'cuz': 'E:\CUZ2021',
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'kadid': list(range(0, 140000)),
        'cuz': list(range(0, 240000)),
        'cuzG': list(range(0, 44000)),
        'cuzP': list(range(0, 36000)),
        'cuzE': list(range(0, 3000)),
        'cuzEG': list(range(0, 799)),
        'cuzEP': list(range(0, 798)),

    }
    idx_config = {}

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        sel_num_cuz = [xx for xx in img_num['cuz']]
        sel_num_cuzG = [xx for xx in img_num['cuzG']]
        sel_num_cuzP = [xx for xx in img_num['cuzP']]
        sel_num_cuzE = [xx for xx in img_num['cuzE']]
        sel_num_cuzEG = [xx for xx in img_num['cuzEG']]
        sel_num_cuzEP = [xx for xx in img_num['cuzEP']]

        sel_num_newT = [xx for xx in img_num[config.datasetT]]
        random.shuffle(sel_num_cuz)
        random.shuffle(sel_num_cuzG)
        random.shuffle(sel_num_cuzP)
        random.shuffle(sel_num_cuzE)
        random.shuffle(sel_num_cuzEG)
        random.shuffle(sel_num_cuzEP)
        idx_config['cuz'] = sel_num_cuz
        idx_config['cuzG'] = sel_num_cuzG
        idx_config['cuzP'] = sel_num_cuzP
        idx_config['cuzE'] = sel_num_cuzE
        idx_config['cuzEtr'] = sel_num_cuzE[:round(0.8*3000)]
        idx_config['cuzEte'] = sel_num_cuzE[round(0.8*3000):]
        idx_config['cuzEG'] = sel_num_cuzEG
        idx_config['cuzEP'] = sel_num_cuzEP
        idx_config['test'] = sel_num_cuzE[round(0.8*3000):]
        solver = HyperIQASolver(config, idx_config)
        solver.train()

    print('End.....')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='kadid',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--datasetT', dest='datasetT', type=str, default='tid',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=5,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Epochs for training')
    parser.add_argument('--rounds', dest='rounds', type=int, default=30000, help='Rounds for training')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    config = parser.parse_args()
    main(config)
