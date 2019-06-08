#!/bin/bash3

python main.py --data_path ../processed/ --saved ./saved_models/sn/checkpoint_temp_ep3_bn_0.3_MSE.pth.tar --lr 0.0001 --bn_momentum 0.30 --criterion MSE --bn_warmup false
# python main.py --data_path ../processed/ --reset --workers 16 --lr 0.05 --bn_momentum 0.10
# python main.py --data_path ../processed/ --reset --workers 16 --lr 0.05 --bn_momentum 0.30
# python main.py --data_path ../processed/ --reset --workers 16 --lr 0.05 --bn_momentum 0.40
# python main.py --data_path ../processed/ --reset --workers 16 --lr 0.05 --bn_momentum 0.50
