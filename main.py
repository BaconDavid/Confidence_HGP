# %%
from model import resnet
from model import densenet_BC
from model import vgg
from model.Swin_TS import Swin3DTransformer
from monai.optimizers import WarmupCosineSchedule
# %%
import data as dataset
import crl_utils
import metrics
import utils
import train

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import random
import numpy as np
# %%
parser = argparse.ArgumentParser(description='Confidence Aware Learning')
parser.add_argument('--epochs', default=1, type=int, help='Total number of epochs to run')
parser.add_argument('--data', default='HGP', type=str, help='Dataset name to use [cifar10, cifar100, svhn]')
parser.add_argument('--model', default='Res', type=str, help='Models name to use [res, dense, vgg]')
parser.add_argument('--rank_target', default='softmax', type=str, help='Rank_target name to use [softmax, margin, entropy]')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--data_path', default='../Data/Mixed_HGP/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='../Data/test/', type=str, help='Savefiles directory')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fold', default=0, type=int, help='which fold to train')
parser.add_argument('--exp_name', default='Confi_lg_tm_plus_5_0', type=str, help='experiment name')
parser.add_argument('--device', default='cpu', type=str, help='device to use')
parser.add_argument('--train_data_folds', default='../Data/Mixed_HGP_Folds/', type=str, help='train data label ')
args = parser.parse_args()

def set_seed(seed=114514):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()
    # set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    # check save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_data_label = args.train_data_folds + 'train_cv_' + str(args.fold) + '.csv'
    test_data_label = args.train_data_folds + 'val_cv_' + str(args.fold) + '.csv'


    # make dataloader
    train_loader, test_loader, \
    test_onehot, test_label = dataset.get_loader(args.data,
                                                 args.data_path,
                                                 args.model,
                                                 train_data_label,
                                                 test_data_label,
                                                 label_name='HGP_Type')
                                                 

    # set num_class
    if args.data == 'cifar100':
        num_class = 100
    else:
        num_class = 10

    # set num_classes
    resnet_model_dict = {
        "num_classes": 2,
        "n_input_channels": 1,
        "widen_factor": 1
    }

    Swin_model_dict = {
        "img_size": (32,256,256),
        "in_channels": 1,
        "num_class": 2,
        "num_heads": [3,6,12,24],
        "out_channels": 1,
        "depths": (2,2,2,2)
    }

    # set model
    if args.model == 'Res':
        model = resnet.resnet10(**resnet_model_dict).to(args.device)
    elif args.model == 'Swin':
        model = Swin3DTransformer(**Swin_model_dict).to(args.device)


    # set criterion
    cls_criterion = nn.CrossEntropyLoss().to(args.device)
    ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(args.device)

    # set optimizer (default:sgd)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.0001,
                          momentum=0.9,
                          weight_decay=0.0,
                          nesterov=False)

    scheduler = WarmupCosineSchedule(optimizer,
                                    200,
                                    15000)
    # make logger
    train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
    result_logger = utils.Logger(os.path.join(save_path, 'result.log'))

    # make History Class
    correctness_history = crl_utils.History(len(train_loader.dataset))

    # start Train
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train.train(train_loader,
                    model,
                    cls_criterion,
                    ranking_criterion,
                    optimizer, 
                    epoch,
                    correctness_history,
                    train_logger,
                    args)

        # save model
        if epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(save_path, 'model.pth'))
    # finish train

    # calc measure
    acc, aurc, eaurc, aupr, fpr, ece, nll, brier = metrics.calc_metrics(test_loader,
                                                                        test_label,
                                                                        test_onehot,
                                                                        model,
                                                                        cls_criterion,
                                                                        device=args.device)
    acc, softmax_lst, correct_lst, logit_lst = metrics.save_output(test_loader,test_label,test_onehot,model,cls_criterion,device=args.device)

    softmax_array = np.array(softmax_lst)
    correct_array = np.array(correct_lst)
    logit_array = np.array(logit_lst)

    #save output
    save_path = args.save_path + args.exp_name + '/' + str(args.fold) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path,'softmax_array.npy'), softmax_array)
    np.save(os.path.join(save_path, 'correct_array.npy'), correct_array)
    np.save(os.path.join(save_path, 'logit_array.npy'), logit_array)

    # result write
    result_logger.write([acc, aurc*1000, eaurc*1000, aupr*100, fpr*100, ece*100, nll*10, brier*100])
    
if __name__ == "__main__":
    if args.data != 'HGP':
        main()

    else:
        
        print('successfully load the config file !')
        set_seed(114514)
        main()




# %%
