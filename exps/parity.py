#!/usr/bin/env python3

import argparse

import os
import sys
import csv
import shutil

import numpy.random as npr

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import satnet
from tqdm.auto import tqdm

class CSVLogger(object):
    def __init__(self, fname):
        self.f = open(fname, 'w')
        self.logger = csv.writer(self.f)

    def log(self, fields):
        self.logger.writerow(fields)
        self.f.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='parity')
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--batchSz', type=int, default=100)
    parser.add_argument('--testBatchSz', type=int, default=500)
    parser.add_argument('--nEpoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--seq', type=int, default=20)
    parser.add_argument('--save', type=str)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--aux', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--adam', action='store_true')

    args = parser.parse_args()

    # For debugging: fix the random seed
    npr.seed(1)
    torch.manual_seed(7)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda: 
        print('Using', torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()

    save = 'parity.aux{}-m{}-lr{}-bsz{}'.format(
            args.aux, args.m, args.lr, args.batchSz)

    if args.save: save = '{}-{}'.format(args.save, save)
    save = os.path.join('logs', save)
    if os.path.isdir(save): shutil.rmtree(save)
    os.makedirs(save)

    L = args.seq

    with open(os.path.join(args.data_dir, str(L), 'features.pt'), 'rb') as f:
        X = torch.load(f).float()
    with open(os.path.join(args.data_dir, str(L), 'labels.pt'), 'rb') as f:
        Y = torch.load(f).float()

    if args.cuda: X, Y = X.cuda(), Y.cuda()

    N = X.size(0)

    nTrain = int(N*(1-args.testPct))
    nTest = N-nTrain

    assert(nTrain % args.batchSz == 0)
    assert(nTest % args.testBatchSz == 0)

    train_is_input = torch.IntTensor([1,1,0]).repeat(nTrain,1)
    test_is_input = torch.IntTensor([1,1,0]).repeat(nTest,1)
    if args.cuda: train_is_input, test_is_input = train_is_input.cuda(), test_is_input.cuda()

    train_set = TensorDataset(X[:nTrain], train_is_input, Y[:nTrain])
    test_set =  TensorDataset(X[nTrain:], test_is_input, Y[nTrain:])

    model = satnet.SATNet(3, args.m, args.aux, prox_lam=1e-1)
    if args.cuda: model = model.cuda()

    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_logger = CSVLogger(os.path.join(save, 'train.csv'))
    test_logger = CSVLogger(os.path.join(save, 'test.csv'))
    fields = ['epoch', 'loss', 'err']
    train_logger.log(fields)
    test_logger.log(fields)

    test(0, model, optimizer, test_logger, test_set, args.testBatchSz)
    for epoch in range(1, args.nEpoch+1):
        train(epoch, model, optimizer, train_logger, train_set, args.batchSz)
        test(epoch, model, optimizer, test_logger, test_set, args.testBatchSz)

def apply_seq(net, zeros, batch_data, batch_is_inputs, batch_targets):
    y = torch.cat([batch_data[:,:2], zeros], dim=1)
    y = net(y, batch_is_inputs)
    L = batch_data.size(1)
    for i in range(L-2):
        y = torch.cat([y[:,-1].unsqueeze(1), batch_data[:,i+2].unsqueeze(1), zeros], dim=1)
        y = net(((y-0.5).sign()+1)/2, batch_is_inputs)
    loss = F.binary_cross_entropy(y[:,-1], batch_targets[:,-1])
    return loss, y

def run(epoch, model, optimizer, logger, dataset, batchSz, to_train):
    loss_final, err_final = 0, 0

    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))
        
    start = torch.zeros(batchSz, 1)
    if next(model.parameters()).is_cuda: start = start.cuda()

    for i,(data,is_input, label) in tloader:
        if to_train: optimizer.zero_grad()

        loss, pred = apply_seq(model, start, data, is_input, label)

        if to_train:
            loss.backward()
            optimizer.step()

        err = computeErr(pred, label)
        tloader.set_description('Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(
            epoch, ('Train' if to_train else 'Test '), loss.item(), err))
        loss_final += loss.item()
        err_final += err

    loss_final, err_final = loss_final/len(loader), err_final/len(loader)
    logger.log((epoch, loss_final, err_final))

    if not to_train:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_final, err_final))

def train(epoch, model, optimizer, logger, dataset, batchSz):
    run(epoch, model, optimizer, logger, dataset, batchSz, True)

@torch.no_grad()
def test(epoch, model, optimizer, logger, dataset, batchSz):
    run(epoch, model, optimizer, logger, dataset, batchSz, False)

@torch.no_grad()
def computeErr(pred, target):
    y = (pred[:,-1]-0.5)
    t = (target[:,-1]-0.5)
    correct = ((y * t).sign()+1.)/2
    acc = correct.sum().float()/target.size(0)

    return 1-float(acc)

if __name__ == '__main__':
    main()
