#!/bin/bash

python main.py --data_path ../processed/ --saved_model ./saved_models/sn/best_checkpoint.pth.tar --workers 2 --lr 0.05
