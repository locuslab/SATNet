#!/usr/bin/env python3
#
# Partly derived from:
#   https://github.com/locuslab/optnet/blob/master/sudoku/train.py 

import argparse

import os
import shutil
import csv

import numpy as np
import numpy.random as npr
#import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import satnet

class SudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m):
        super(SudokuSolver, self).__init__()
        n = boardSz**6
        self.sat = satnet.SATNet(n, m, aux)

    def forward(self, y_in, mask):
        out = self.sat(y_in, mask)
        return out

class DigitConv(nn.Module):
    '''
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    '''
    def __init__(self):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)[:,:9].contiguous()

class MNISTSudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m):
        super(MNISTSudokuSolver, self).__init__()
        self.digit_convnet = DigitConv()
        self.sudoku_solver = SudokuSolver(boardSz, aux, m)
        self.boardSz = boardSz
        self.nSq = boardSz**2
    
    def forward(self, x, is_inputs):
        nBatch = x.shape[0]
        x = x.flatten(start_dim = 0, end_dim = 1)
        digit_guess = self.digit_convnet(x)
        puzzles = digit_guess.view(nBatch, self.nSq * self.nSq * self.nSq)

        solution = self.sudoku_solver(puzzles, is_inputs)
        return solution

class CSVLogger(object):
    def __init__(self, fname):
        self.f = open(fname, 'w')
        self.logger = csv.writer(self.f)

    def log(self, fields):
        self.logger.writerow(fields)
        self.f.flush()

class FigLogger(object):
    def __init__(self, fig, base_ax, title):
        self.colors = ['tab:red', 'tab:blue']
        self.labels = ['Loss (entropy)', 'Error']
        self.markers = ['d', '.']
        self.axes = [base_ax, base_ax.twinx()]
        base_ax.set_xlabel('Epochs')
        base_ax.set_title(title)
        
        for i, ax in enumerate(self.axes):
            ax.set_ylabel(self.labels[i], color=self.colors[i])
            ax.tick_params(axis='y', labelcolor=self.colors[i])

        self.reset()
        self.fig = fig
        
    def log(self, args):
        for i, arg in enumerate(args[-2:]):
            self.curves[i].append(arg)
            x = list(range(len(self.curves[i])))
            self.axes[i].plot(x, self.curves[i], self.colors[i], marker=self.markers[i])
            self.axes[i].set_ylim(0, 1.05)
            
        self.fig.canvas.draw()
        
    def reset(self):
        for ax in self.axes:
            for line in ax.lines:
                line.remove()
        self.curves = [[], []]

def print_header(msg):
    print('===>', msg)

def find_unperm(perm):
    unperm = torch.zeros_like(perm)
    for i in range(perm.size(0)):
        unperm[perm[i]] = i
    return unperm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='sudoku')
    parser.add_argument('--boardSz', type=int, default=3)
    parser.add_argument('--batchSz', type=int, default=40)
    parser.add_argument('--testBatchSz', type=int, default=40)
    parser.add_argument('--aux', type=int, default=300)
    parser.add_argument('--m', type=int, default=600)
    parser.add_argument('--nEpoch', type=int, default=100)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--save', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--perm', action='store_true')

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

    save = 'sudoku{}{}.boardSz{}-aux{}-m{}-lr{}-bsz{}'.format(
            '.perm' if args.perm else '', '.mnist' if args.mnist else '',
            args.boardSz, args.aux, args.m, args.lr, args.batchSz)
    if args.save: save = '{}-{}'.format(args.save, save)
    save = os.path.join('logs', save)
    if os.path.isdir(save): shutil.rmtree(save)
    os.makedirs(save)

    #setproctitle.setproctitle('sudoku.{}'.format(save))

    print_header('Loading data')

    with open(os.path.join(args.data_dir, 'features.pt'), 'rb') as f:
        X_in = torch.load(f)
    with open(os.path.join(args.data_dir, 'features_img.pt'), 'rb') as f:
        Ximg_in = torch.load(f)
    with open(os.path.join(args.data_dir, 'labels.pt'), 'rb') as f:
        Y_in = torch.load(f)
    with open(os.path.join(args.data_dir, 'perm.pt'), 'rb') as f:
        perm = torch.load(f)

    N = X_in.size(0)
    nTrain = int(N*(1.-args.testPct))
    nTest = N-nTrain
    assert(nTrain % args.batchSz == 0)
    assert(nTest % args.testBatchSz == 0)

    print_header('Forming inputs')
    X, Ximg, Y, is_input = process_inputs(X_in, Ximg_in, Y_in, args.boardSz)
    data = Ximg if args.mnist else X
    if args.cuda: data, is_input, Y = data.cuda(), is_input.cuda(), Y.cuda()

    unperm = None
    if args.perm and not args.mnist:
        print('Applying permutation')
        data[:,:], Y[:,:], is_input[:,:] = data[:,perm], Y[:,perm], is_input[:,perm]
        unperm = find_unperm(perm)

    train_set = TensorDataset(data[:nTrain], is_input[:nTrain], Y[:nTrain])
    test_set =  TensorDataset(data[nTrain:], is_input[nTrain:], Y[nTrain:])

    print_header('Building model')
    if args.mnist:
        model = MNISTSudokuSolver(args.boardSz, args.aux, args.m)
    else:
        model = SudokuSolver(args.boardSz, args.aux, args.m)

    if args.cuda: model = model.cuda()

    if args.mnist:
        optimizer = optim.Adam([
            {'params': model.sudoku_solver.parameters(), 'lr': args.lr},
            {'params': model.digit_convnet.parameters(), 'lr': 1e-5},
            ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.model:
        model.load_state_dict(torch.load(args.model))

    train_logger = CSVLogger(os.path.join(save, 'train.csv'))
    test_logger = CSVLogger(os.path.join(save, 'test.csv'))
    fields = ['epoch', 'loss', 'err']
    train_logger.log(fields)
    test_logger.log(fields)

    test(args.boardSz, 0, model, optimizer, test_logger, test_set, args.testBatchSz, unperm)
    for epoch in range(1, args.nEpoch+1):
        train(args.boardSz, epoch, model, optimizer, train_logger, train_set, args.batchSz, unperm)
        test(args.boardSz, epoch, model, optimizer, test_logger, test_set, args.testBatchSz, unperm)
        #torch.save(model.state_dict(), os.path.join(save, 'it'+str(epoch)+'.pth'))

def process_inputs(X, Ximg, Y, boardSz):
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()

    Ximg = Ximg.flatten(start_dim=1, end_dim=2)
    Ximg = Ximg.unsqueeze(2).float()

    X      = X.view(X.size(0), -1)
    Y      = Y.view(Y.size(0), -1)
    is_input = is_input.view(is_input.size(0), -1)

    return X, Ximg, Y, is_input

def run(boardSz, epoch, model, optimizer, logger, dataset, batchSz, to_train=False, unperm=None):

    loss_final, err_final = 0, 0

    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))

    for i,(data,is_input,label) in tloader:
        if to_train: optimizer.zero_grad()
        preds = model(data.contiguous(), is_input.contiguous())
        loss = nn.functional.binary_cross_entropy(preds, label)

        if to_train:
            loss.backward()
            optimizer.step()

        err = computeErr(preds.data, boardSz, unperm)/batchSz
        tloader.set_description('Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err))
        loss_final += loss.item()
        err_final += err

    loss_final, err_final = loss_final/len(loader), err_final/len(loader)
    logger.log((epoch, loss_final, err_final))

    if not to_train:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_final, err_final))

    #print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()

def train(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, True, unperm)

@torch.no_grad()
def test(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, False, unperm)

@torch.no_grad()
def computeErr(pred_flat, n, unperm):
    if unperm is not None: pred_flat[:,:] = pred_flat[:,unperm]

    nsq = n ** 2
    pred = pred_flat.view(-1, nsq, nsq, nsq)

    batchSz = pred.size(0)
    s = (nsq-1)*nsq//2 # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq-1)
        valid *= (x.sum(1) == s)
        return valid.bitwise_not()

    boardCorrect = torch.ones(batchSz).type_as(pred)
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:,j,:])] = 0
        boardCorrect[invalidGroups(I[:,:,j])] = 0

        # Check the jth block.
        row, col = n*(j // n), n*(j % n)
        M = invalidGroups(I[:,row:row+n,col:col+n].contiguous().view(batchSz,-1))
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return float(batchSz-boardCorrect.sum())

if __name__=='__main__':
    main()
