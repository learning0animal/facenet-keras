# -*- coding: utf-8 -*-

import json
# import argparse
from keras import optimizers
from loader import TripDataGenerator
from model import build_trip_model
from loss import triplet_loss


def main():
     
    model = build_trip_model()
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=triplet_loss)
    print('[INFO] init model')
    
    TRN_PATH = './data/train'
    TST_PATH = './data/test'
    trn_gen = TripDataGenerator(TRN_PATH)
    val_gen = TripDataGenerator(TST_PATH)

    model.fit_generator(trn_gen, steps_per_epoch=trn_gen.n // trn_gen.batch_size, 
                    validation_data=val_gen, validation_steps=val_gen.n // val_gen.batch_size, 
                    epochs=1)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, help='network config', default='net_cfg.json')
    # args = parser.parse_args()

    main()    