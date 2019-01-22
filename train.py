# -*- coding: utf-8 -*-

import json
# import argparse
from keras import optimizers
from loader import TripDataGenerator
from model import build_trip_model
from loss import triplet_loss
from config import config


# def load_config(config_path):
#     with open(config_path, 'rt') as f:
#         config = json.loads(f.read())
#     return config


def main():
    # config_path = args.config
    # config = load_config(config_path)
    # print('[INFO] loaded config')
     
    model = build_trip_model(config['img_h'], config['img_w'], config['n_channels'], config['embedding_size'])
    model.compile(optimizer='adam', loss=triplet_loss)
    print('[INFO] init model')
    
    TRN_PATH = './data/train'
    TST_PATH = './data/test'
    trn_gen = TripDataGenerator(TRN_PATH, config)
    val_gen = TripDataGenerator(TST_PATH, config)

    model.fit_generator(trn_gen, steps_per_epoch=trn_gen.n // trn_gen.batch_size, 
                    validation_data=val_gen, validation_steps=val_gen.n // val_gen.batch_size, 
                    epochs=1)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, help='network config', default='net_cfg.json')
    # args = parser.parse_args()

    main()    