#!/bin/bash

python main.py --data_path ../processed/ --saved ./saved_models/sn/checkpoint_ep0_bn_0.3.pth.tar --workers 16 --lr 0.05 --bn_momentum 0.30 --criterion MSE
# python main.py --data_path ../processed/ --reset --workers 16 --lr 0.05 --bn_momentum 0.10
# python main.py --data_path ../processed/ --reset --workers 16 --lr 0.05 --bn_momentum 0.30
# python main.py --data_path ../processed/ --reset --workers 16 --lr 0.05 --bn_momentum 0.40
# python main.py --data_path ../processed/ --reset --workers 16 --lr 0.05 --bn_momentum 0.50
