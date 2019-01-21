# -*- coding: utf-8 -*-

import json
import argparse
from keras import optimizers
from model import build_trip_model
from loss import triplet_loss


def load_config(config_path):
    with open(config_path, 'rt') as f:
        config = json.loads(f.read())
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='network config', default='net_cfg.json')
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    print('[INFO] loaded config')
     
    model = build_trip_model(config['img_h'], config['img_w'], config['n_channels'], config['embedding_size'])
    print('[INFO] init model')
