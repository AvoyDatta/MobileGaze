import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import thop

from ITrackerData import ITrackerData
from MobileGaze import MobileGaze

'''
Train/test code for MobileGaze.
Code adapted from iTracker

Author: Avoy Datta (avoy.datta@stanford.edu)

'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='MobileGaze-pytorch-Trainer.')
parser.add_argument('--saved_model', type=str, default=None, help="Path to saved model.")
parser.add_argument('--data_path', help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.")
parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False, help="Test mode.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
parser.add_argument('--workers', type=int, nargs='?', const=True, default=2, help="Number of CPU cores.")

args = parser.parse_args()

# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.test # Only run test, no training

workers = args.workers
num_gpus = 0 if not torch.cuda.is_available() else torch.cuda.device_count() 
print("Using {} GPUs, {} CPU workers.".format(num_gpus, workers))
epochs = 25
batch_per_gpu = 100
batch_size = torch.cuda.device_count()*batch_per_gpu # Change if out of cuda memory

#Hyperparams used in iTracker 
base_lr = 0.01
adam_betas = [0.9, .999]
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr
save_every = 1
lr_decay = 0.4
lr_period = 10

CHECKPOINTS_PATH = './saved_models/sn/'

#Note that --reset overrides this
saved_model_path = None if not args.saved_model else str(args.saved_model)

def main():
    global args, best_prec1, weight_decay, momentum

    model = MobileGaze().cuda()
    
    print("Total number of model parameters: ", sum(p.numel() for p in model.parameters()))
    print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model = torch.nn.DataParallel(model)

    imSize=(224,224)
    cudnn.benchmark = True   

    epoch = 0
    if doLoad:
        saved = load_checkpoint(saved_model_path)
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    
    dataTrain = ITrackerData(dataPath = args.data_path, split='train', imSize = imSize)
    dataVal = ITrackerData(dataPath = args.data_path, split='val', imSize = imSize)
    dataTest = ITrackerData(dataPath = args.data_path, split='test', imSize = imSize)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataTest,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    
    print ("Size of training, validation and test sets: {}, {}, {}".format(batch_size*len(train_loader), batch_size*len(val_loader), batch_size*len(test_loader)))
    # Test mode
    if doTest: 
        test_mean = validate(test_loader, model, criterion, epoch)
        print(test_mean)
        return
   
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr,
                                betas=adam_betas)
        
    train_model(model, train_loader, val_loader, optimizer, criterion, epoch, epochs)
    

def train_model(model, train_loader, val_loader, optimizer, criterion, start_epoch, epochs):
    
    loss_list = []
    for epoch in range(start_epoch, epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        print("Learning rate for epoch{}: {}".format(epoch, lr))
        
        # train for one epoch
        train_epoch(model, train_loader, optimizer, criterion, epoch, loss_list)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_name = 'checkpoint_ep{}.pth.tar'.format(epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, )
        
    np.save("./train_losses.npy", np.array(losses))
    
def train_epoch(model, train_loader, optimizer, criterion, epoch, loss_list):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        time0 = time.time()
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
#         imFace = torch.autograd.Variable(imFace, requires_grad = False)
#         imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
#         imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
#         faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
#         gaze = torch.autograd.Variable(gaze, requires_grad = False)
        time1 = time.time() - time0
        print("Time for sec 1 {}".format(time1))
        time1 = time.time()
        # compute output
#         model_stats = dict()
#         with torch.no_grad():
#         if i == 0:          
        output = model(imFace, imEyeL, imEyeR, faceGrid)
#                 print("Number of parameters %i, number of flops: %i" % (model_stats['params'], model_stats['flops']))
#         else:      
#             output = model(imFace, imEyeL, imEyeR, faceGrid)

        time2 = time.time() - time1
        print("Time for sec 2 {}".format(time2))
        time2 = time.time()
        loss = criterion(output, gaze)
        losses.update(loss.data.item(), imFace.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

#         count=count+1
        time3 = time.time() - time2
        print("Time for sec 3 {}".format(time3))

        if i % print_freq == 0:
            print('Epoch (train): [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))
            loss_list.append(losses.val)
            
def validate(val_loader, model, criterion, epoch):
    count_test = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        model_stats = dict()
        with torch.no_grad():
            if i == 0:          
                output = model(imFace, imEyeL, imEyeR, faceGrid, model_stats=model_stats)
                print("Number of parameters %i, number of flops: %i" % (model_stats['params'], model_stats['flops']))
            else:      
                output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item(), imFace.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))

    return lossesLin.avg


def load_checkpoint(filepath='./sn/best_checkpoint.pth.tar'):
#     filename = os.path.join(CHECKPOINTS_PATH, filename)
    print("Loading from: ", filepath)
    if not os.path.isfile(filepath):
        return None
    state = torch.load(filepath)
    return state

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_checkpoint.pth.tar')
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename) #overwrites existing file


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by lr_decay every lr_period epochs"""
    lr = base_lr * (lr_decay ** (epoch // lr_period))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
    return lr

if __name__ == "__main__":
    main()
    print('DONE')
