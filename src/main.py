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
# import tensorboard 
from tensorboardX import SummaryWriter
from PIL import Image

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
parser.add_argument('--workers', type=int, nargs='?', const=True, help="Number of CPU cores.")
parser.add_argument('--lr', type=float, nargs='?', const=True, default=0.01, help="Initial learning rate.")
parser.add_argument('--bn_momentum', type=float, nargs='?', const=True, default=0.25, help="Momentum for batch normalization.")
parser.add_argument('--criterion', type=str, nargs='?', const=True, default='MSE', help="Loss function used.")
parser.add_argument('--bn_warmup', type=str, nargs='?', const=True, default='false', help="Warmup==true for BN at test-time.")


args = parser.parse_args()

bn_warmup = str2bool(args.bn_warmup) 
# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.test # Only run test, no training

num_gpus = 0 if not torch.cuda.is_available() else torch.cuda.device_count() 
workers = args.workers if args.workers else int(4 * num_gpus) 

print("Using {} GPUs, {} CPU workers.".format(num_gpus, workers))
epochs = 25
batch_per_gpu = 64
batch_size = torch.cuda.device_count()*batch_per_gpu # Change if out of cuda memory

val_batch = num_gpus * 16

base_lr = args.lr
adam_betas = [0.9, .999]
print_freq = 10
prec1 = 0
lr = base_lr
save_every = 200 #num of iters to save temp copy
log_every=20
log_dir = '../tb_logs/sn/' #Unused
lr_decay = 0.5
lr_period = 5
bn_momentum = args.bn_momentum
criterion_name = args.criterion
epoch_divide = 0.5 #Train over 1 epoch divided by this value

CHECKPOINTS_PATH = './saved_models/sn/'

#Note that --reset overrides this
saved_model_path = None if not args.saved_model else str(args.saved_model)
log_param_freq = 4 #Number of times per epoch to load params

writer = SummaryWriter()

outlier_factor = 2.0
# best_prec1 = int(1e20)


def main():
#     global args, best_prec1, weight_decay, momentum

    model = MobileGaze(bn_momentum).cuda()
    for name, value in model.named_parameters():
        print("({}, {})".format(name, value.requires_grad))
        
    print("Total number of model parameters: ", sum(p.numel() for p in model.parameters()))
    print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model = torch.nn.DataParallel(model)

    imSize=(224,224)
    cudnn.benchmark = True   

    epoch = 0
    best_prec1 = int(1e20)

    if doLoad:
        saved = load_checkpoint(saved_model_path)
        if saved:
            print('Loading checkpoint for epoch %05d with l2 error %.5f (which is the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch'] + 1
#             if 'temp' in saved_model_path: epoch -= 1 #Model loaded from emergency ckpt
            best_prec1 = saved['best_prec1']
                
        else:
            print('Warning: Could not read checkpoint!')
    else:
        print("Training model from scratch.")
    
    dataTrain = ITrackerData(dataPath = args.data_path, split='train', imSize = imSize)
    dataVal = ITrackerData(dataPath = args.data_path, split='val', imSize = imSize)
    dataTest = ITrackerData(dataPath = args.data_path, split='test', imSize = imSize)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=val_batch, shuffle=True,
        num_workers=workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataTest,
        batch_size=val_batch, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    
    print ("Size of training, validation and test sets: {}, {}, {}".format(batch_size*len(train_loader), val_batch*len(val_loader), val_batch*len(test_loader)))
    
    criterion = nn.SmoothL1Loss().cuda() if criterion_name == 'huber' else nn.MSELoss().cuda()
    
    print("Criterion used: {}".format(criterion_name))
    
    # Test mode
    if doTest: 
        test_mean = validate(test_loader, model, criterion, epoch)
        print(test_mean)
        return

    optimizer = torch.optim.Adam(model.parameters(), base_lr, betas=adam_betas)
     
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l2_errors = AverageMeter()
    loss_list = []
    start_epoch=epoch
    for epoch in range(start_epoch, epochs):
        lr = adjust_learning_rate(optimizer, epoch, base_lr)
        print("Learning rate for epoch{}: {}".format(epoch, lr))
        
        # train for one epoch
        try: 
            train_epoch(model, train_loader, optimizer, criterion, epoch, loss_list,
                       batch_time, data_time, losses, l2_errors)

        except KeyboardInterrupt: 
            pass
        
        save_name = 'checkpoint_ep{}_bn_{}_{}.pth.tar'.format(epoch, bn_momentum, criterion_name)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, False, save_name)
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_name = 'checkpoint_ep{}_bn_{}_{}.pth.tar'.format(epoch, bn_momentum, criterion_name)
        if is_best:
            save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, True, save_name)
    
    np.save("./losses_lr{}_ep{}.npy".format(lr, epochs), np.array(loss_list))

    
    # export scalar data to JSON for external processing
#     writer.export_scalars_to_json("./all_scalars.json")
    writer.add_scalar('bn_momentum', bn_momentum, 0)

    writer.close()

def train_epoch(model, train_loader, optimizer, criterion, epoch, loss_list, 
                batch_time, data_time, losses, l2_errors):
    
    train_counter = 0
    global_counter = epoch*epoch_divide*len(train_loader) + train_counter
    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, imFacePath) in enumerate(train_loader):
#         if (i > epoch_divide * len(train_loader)):
#             break
        # measure data loading time
#         if (i > 1000): break
        data_time.update(time.time() - end)
        time0 = time.time()
        imFace = imFace.cuda(non_blocking=True)
        imEyeL = imEyeL.cuda(non_blocking=True)
        imEyeR = imEyeR.cuda(non_blocking=True)
        faceGrid = faceGrid.cuda(non_blocking=True)
        gaze = gaze.cuda(non_blocking=True)
#         imFace = torch.autograd.Variable(imFace, requires_grad = False)
#         imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
#         imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
#         faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
#         gaze = torch.autograd.Variable(gaze, requires_grad = False)
        time1 = time.time() - time0
#         print("Time for sec 1 {}".format(time1))
        time1 = time.time()
        # compute output
#         model_stats = dict()

#         if i == 0:          
        output = model(imFace, imEyeL, imEyeR, faceGrid)
#                 print("Number of parameters %i, number of flops: %i" % (model_stats['params'], model_stats['flops']))
#         else:      
#             output = model(imFace, imEyeL, imEyeR, faceGrid)

        time2 = time.time() - time1
#         print("Time for sec 2 {}".format(time2))
        time2 = time.time()
#         print("output:\n", str(output.detach().cpu().numpy()))
#         print("gaze:\n", str(gaze.detach().cpu().numpy()))
        loss = criterion(output, gaze)
        losses.update(loss.data.item(), imFace.size(0))

        l2_error = output - gaze
        l2_error = torch.mul(l2_error,l2_error)
        l2_error = torch.sum(l2_error,1)
        l2_error = torch.mean(torch.sqrt(l2_error)) 
            
#         if (l2_error.item() > outlier_factor * l2_errors.avg) and i > 200: #Ignore Outlier
#             print("Minibatch {} of epoch {} skipped during training.".format(train_counter, epoch))
#             for img_idx in range(4):
#                 imFace = Image.open(str(imFacePath[img_idx])).convert('RGB')
#                 img_savepath = "./buggy/train/buggy_epoch{}_iter{}_l2_error{}_{}.jpg".format(epoch, train_counter, l2_error.item(),                                                                            img_idx)
#                 imFace.save(img_savepath, "JPEG")
#         else:
        l2_errors.update(l2_error.item(), imFace.size(0))         
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_counter += 1
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        time3 = time.time() - time2
#         print("Time for sec 3 {}".format(time3))
        if i % save_every == 0:
            save_name = 'checkpoint_temp_ep{}_bn_{}_{}.pth.tar'.format(epoch, bn_momentum, criterion_name)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': l2_error,
            }, False, save_name) #False indicates this should not be saved to best_ckpy
        
        if i % print_freq == 0:
            print('Epoch (train): [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'L2 error{l2_error:.4f}, Avg L2 error {l2_errors.avg:.4f}'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses,
                       l2_error=l2_error.item(), l2_errors=l2_errors))
        
        if i % (log_every * 10) == 0:
            loss_list.append(losses.val)
            
        if i % log_every == 0:
             # ================================================================== #
            #                        TensorboardX Logging                         #
            # ================================================================== #
            
            # 1. Log scalar values (scalar summary)
            info = { 'loss': losses.val, 'loss_avg': losses.avg, 
                     'l2_error': l2_errors.val, 'l2_error_avg': l2_errors.avg, 
                     'batch_time_avg': batch_time.avg, 'data_time_avg': data_time.avg 
                   }

            for tag, value in info.items():
                writer.add_scalar(tag, value, global_counter)            
          
            writer.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)  
            
        del row, imFace, imEyeL, imEyeR, faceGrid, gaze, loss, l2_error
        
    # Log values and gradients of the parameters (histogram summary)
        if i % int(len(train_loader) / log_param_freq) == 0:
            print("Logging param weights and grads...")
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.data.cpu().numpy(), global_counter)
                writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), global_counter)
        
            # 3. Log training images (image summary)
#             info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

#             for tag, images in info.items():
#                 logger.image_summary(tag, images, train_counter)
            
            
            
def validate(val_loader, model, criterion, epoch):
    test_counter=0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    end = time.time()

    warmup_sets = int(len(val_loader) / 20)
    #Stabilize BN values before evaluation
    if (bn_warmup == False):
        print("BN warm-up skipped at test-time")
    else:
        print("Warming up BN metrics")
        for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, imFacePath) in enumerate(val_loader):
#             if (i > warmup_sets): break
            # measure data loading time
            data_time.update(time.time() - end)
            imFace = imFace.cuda()
            imEyeL = imEyeL.cuda()
            imEyeR = imEyeR.cuda()
            faceGrid = faceGrid.cuda()
            gaze = gaze.cuda()
            output = model(imFace, imEyeL, imEyeR, faceGrid)
        print("BN warmed up")

    model.eval()
    oIndex = 0
    
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, imFacePath) in enumerate(val_loader):
        # measure data loading time
#         if i > 200: break
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        # compute output
        model_stats = dict()
        with torch.no_grad():
#             if i == 0:          
            output = model(imFace, imEyeL, imEyeR, faceGrid)
#             print("Number of parameters %i, number of flops: %i" % (model_stats['params'], model_stats['flops']))
#             else:      
#                 output = model(imFace, imEyeL, imEyeR, faceGrid)
        
        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
#         print("output:\n", str(output.detach().cpu().numpy()))
#         print("gaze:\n", str(gaze.cpu().numpy()))
        test_counter+=1
        lossLin = torch.mean(torch.sqrt(lossLin))

#         if (lossLin.item() > outlier_factor * lossesLin.avg) and (i > 5) : #Outlier
#             print("Minibatch {} skipped during validation.".format(i))
#             for img_idx in range(4):
#                 imFace = Image.open(str(imFacePath[img_idx])).convert('RGB')
#                 img_savepath = "./buggy/val/buggy_epoch{}_iter{}_lossLin{}_{}.jpg".format(epoch, i, lossLin.item(),                                                                            img_idx), "JPEG"
#                 imFace.save(img_savepath)
        
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
        if i % log_every == 0:
             # ================================================================== #
            #                        TensorboardX Logging                         #
            # ================================================================== #
            
            # 1. Log scalar values (scalar summary)
            info = { 'val_loss_epoch{}'.format(epoch): losses.val, 'val_loss_avg_epoch{}'.format(epoch): losses.avg,
                    'val_lossLin_epoch{}'.format(epoch): lossesLin.val, 'val_lossLin_avg_epoch{}'.format(epoch): lossesLin.avg}

            for tag, value in info.items():
                writer.add_scalar(tag, value, test_counter)
                          
#         del row, imFace, imEyeL, imEyeR, faceGrid, gaze, loss
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


def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by lr_decay every lr_period epochs"""
    lr = base_lr * (lr_decay ** (epoch // lr_period))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
    return lr

if __name__ == "__main__":
    main()
    print('DONE')

    
# def train_model(model, train_loader, val_loader, optimizer, criterion, start_epoch, epochs,
#                best_prec1):
# #     tb = tensorboard.Tensorboard(log_dir)

#     for epoch in range(start_epoch, epochs):
#         lr = adjust_learning_rate(optimizer, epoch)
#         print("Learning rate for epoch{}: {}".format(epoch, lr))
        
#         # train for one epoch
#         try: 
#             train_epoch(model, train_loader, optimizer, criterion, epoch, loss_list,
#                        batch_time, data_time, losses, l2_errors)

#         except KeyboardInterrupt: 
#             pass
        
#         # evaluate on validation set
#         prec1 = validate(val_loader, model, criterion, epoch)
#         # remember best prec@1 and save checkpoint
#         is_best = prec1 < best_prec1
#         best_prec1 = min(prec1, best_prec1)
#         save_name = 'checkpoint_ep{}_bn_{}_{}.pth.tar'.format(epoch, bn_momentum, criterion_name)
#         save_checkpoint({
#             'epoch': epoch,
#             'state_dict': model.state_dict(),
#             'best_prec1': best_prec1,
#         }, is_best, save_name)
    