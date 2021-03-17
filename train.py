#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:42:21 2019

@author: carsault
"""

#%%
import argparse
import torch
from torch.utils import data
import torch.nn as nn
import os, errno
from utilities import dataImport
from utilities import chordUtil
from utilities import modelsGen
from utilities import transformer
from utilities.chordUtil import *
from utilities.transformer import *
from seq2seq import seq2seqModel
from seq2seq.seq2seqModel import *
from utilities import util
from utilities.util import *
from utilities import loss as lossPerp
from utilities import testFunc
from utilities.testFunc import *
import pickle


import torch.nn.functional as F
from torch.autograd import Variable
#%%
"""
###################

Argument parsing

###################
"""
parser = argparse.ArgumentParser(description='Hierarchical Latent Space')
# General
parser.add_argument('--dataFolder',   type=str,   default='a0_124_2',    help='name of the data folder')
parser.add_argument('--batch_size',      type=int,   default="500",                                help='batch size (default: 50)')
parser.add_argument('--alpha',      type=str,   default='a0',                            help='type of alphabet')
parser.add_argument('--lenSeq',      type=int,   default= 8,                            help='length of input sequence')
parser.add_argument('--lenPred',      type=int,   default=8,                            help='length of predicted sequence')
parser.add_argument('--decimList', nargs="+",     type=int,   default=[1],                            help='list of decimations (default: [1])')
parser.add_argument('--latent',     type=int,   default=50,                                 help='size of the latent space (default: 50)')
parser.add_argument('--hidden',     type=int,   default=500,                                 help='size of the hidden layer (default: 500)')
parser.add_argument('--modelType',      type=str,   default='mlpDecim',                            help='type of model to evaluate')
parser.add_argument('--layer',     type=int,   default=1,                                 help='number of the hidden layer - 2 (default: 1)')
parser.add_argument('--dropRatio',     type=float,   default=0.5,                                 help='drop Out ratio (default: 0.5)')
parser.add_argument('--device',     type=str,   default="cuda",                              help='set the device (cpu or cuda, default: cpu)')
parser.add_argument('--epochs',     type=int,   default=20000,                                help='number of epochs (default: 15000)')
parser.add_argument('--lr',         type=float, default=1e-4,                               help='learning rate for Adam optimizer (default: 2e-4)')
parser.add_argument('--random_state',   type=int,   default=2,    help='seed for the random train/test split')
# RNN Learning
parser.add_argument('--teacher_forcing_ratio',     type=float,   default=0.5,                                 help='between 0 and 1 (default: 0.5)')
parser.add_argument('--professor_forcing',     type=util.str2bool,   default=True,                                 help='activate professor forcing GAN training (default: False)')
parser.add_argument('--professor_forcing_ratio',     type=float,   default=0.0,                                 help='between 0 and 1 (default: 0.5)')
parser.add_argument('--attention',     type=util.str2bool,   default=False,                                 help='attention mechanism in LSTM decoder')
parser.add_argument('--expand',     type=util.str2bool,   default=False,                                 help='reduce the latent space in LSTM')
# Save file
parser.add_argument('--foldName',      type=str,   default='modelSave190515',                            help='name of the folder containing the models')
parser.add_argument('--modelName',      type=str,   default='bqwlbq',                            help='name of model to evaluate')
parser.add_argument('--dist',      type=str,   default='euclidian',                            help='distance to compare predicted sequence (default : euclidian')
args = parser.parse_args()
print(args)




#args.dataFolder = args.alpha + "_1_" + str(args.random_state)
args.dataFolder = args.alpha + "_124_" + str(args.random_state)


str1 = ''.join(str(e) for e in args.decimList)

args.modelName = args.dataFolder + "_" + str1 + "_" + args.modelType

# Create save folder
try:
    os.mkdir(args.foldName)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir(args.foldName + '/' + args.modelName)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# CUDA for PyTorch
args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor
'''
if args.device is not torch.device("cpu"):
    print(args.device)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark=True
'''
#%% Dataset 
# Create generators
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 0}

# Create dataset
dataset_train = dataImport.createDatasetFull("datasets/" + args.dataFolder + "/train.pkl")
dataset_valid = dataImport.createDatasetFull("datasets/" + args.dataFolder + "/valid.pkl")
dataset_test = dataImport.createDatasetFull("datasets/" + args.dataFolder + "/test.pkl")

bornInf = {}
bornSup = {}
listNameModel = {}
#listNameModel[2] = "mlp2Decim2bis"
#listNameModel[4] = "mlpDecim4bis"
listNameModel[2] = args.dataFolder + "_2_" + "mlpDecim"
listNameModel[4] = args.dataFolder + "_4_" + "mlpDecim"
res = {}

bornInf[1] = 0
bornSup[1] = 8
bornInf[2] = 8
bornSup[2] = 12
bornInf[4] = 12
bornSup[4] = 14 


    
training_generator = data.DataLoader(dataset_train, pin_memory = True, **params)
validating_generator = data.DataLoader(dataset_valid, pin_memory = True, **params)
testing_generator = data.DataLoader(dataset_test, pin_memory = True, **params)

#%%
maxReps = 2
dictChord, listChord = chordUtil.getDictChordUpgrade(chordUtil.a0, maxReps)
n_categories = len(listChord)
decim = args.decimList[0]
#print(dictChord, listChord)
#print(len(dictChord), len(listChord))
#print(n_categories)
#print(args.hidden, args.latent, decim, args.layer)

#if args.dist != 'None': #this is causing errors w/ new dictionary, correct functions
	#distMat = testFunc.computeMat(dictChord, args.dist)
	#distMat = torch.Tensor(distMat).to(args.device,non_blocking=True)

if args.modelType == "mlpDecim":
    enc = modelsGen.EncoderMLP(args.lenSeq, n_categories, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    #enc = modelsGen.dilatConv(args.lenSeq, 1, n_categories, args.latent)
    dec = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    net = modelsGen.InOutModel(enc,dec)
    criterion = nn.BCELoss()
    if args.decimList[0] != 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

# Print model 
print(net)
#f.write(print(net))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(net))
res['numberOfModelParams'] = count_parameters(net)

if args.device is not "cpu":
    net.to(args.device)

#if   args.modelType != "transformer":
# Choose lose
#if args.decimList[0] != 1:
    #criterion = nn.MSELoss()
    #else:
    #    criterion = nn.BCELoss()
    
# choose optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()



accDiscr = 1
bestAccurValid = 0
accurValid = 0
bestValid_total_loss = 1000

#perp = lossPerp.Perplexity()
# Begin training
for epoch in range(args.epochs):
    print('Epoch number {} '.format(epoch))
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    train_total_loss = 0
    valid_total_loss = 0
    test_total_loss = 0
    for local_batch, local_labels, local_key, local_beat in training_generator:
       # print(local_batch.size())
        #print(local_labels.size())
        #print(local_key, local_key.size())
        #print(local_beat, local_beat.size())
        if len(args.decimList) == 1:
            local_batch = local_batch[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous()
        local_labels = local_labels[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous() 
        local_batch, local_labels = local_batch.to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
        if args.modelType == "mlpDecim":
            net.train() 
            net.zero_grad()
            output = net(local_batch)
            #print(output.size())
            loss = criterion(output, local_labels)

            loss.backward()
            optimizer.step()
            train_total_loss += loss

            
    print(train_total_loss)

    totalVal = 0
    correct = 0
    for local_batch, local_labels, local_key, local_beat in validating_generator:
        if len(args.decimList) == 1:
            local_batch = local_batch[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous()
        local_labels = local_labels[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous() 
        local_batch, local_labels = local_batch.to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True) 
        if args.modelType == "mlpDecim":
            with torch.no_grad():
                net.eval() 
                net.zero_grad()
                output = net(local_batch)
                loss = criterion(output, local_labels)
                valid_total_loss += loss

        if args.decimList[0] == 1:
            #TO MODIFY for accuracy computation adjustment
            for i in range(output.size()[0]):
                
                #old_labels = transfToOld(local_labels[i], lenPred)
               # old_output = transfToOld(output[i], lenPred)
                for j in range(int(args.lenPred/decim)):
                    print(local_labels[i][j].max(0)[1])
                    print(output[i][j].max(0)[1])
                    totalVal += 1
                    result = (output[i][j].max(0)[1] == local_labels[i][j].max(0)[1]).item()
                    correct += result

    if args.decimList[0] == 1:
        accurValid = 100 * correct / totalVal

    if args.decimList[0] == 1:
        if  accurValid > bestAccurValid:
            bestAccurValid = accurValid
            res["params"] = str(args)
            res["bestAccurValid"] = bestAccurValid
            res["epochOnBestAccurValid"] = epoch
            print("new best loss, model saved")
            print('New accuracy of the network on valid dataset: {} %'.format(bestAccurValid))
            torch.save(net.state_dict(), args.foldName + '/' + args.modelName + '/' + args.modelName)
            earlyStop = 0

        else:
            earlyStop += 1
            print("increasing early stopping")


    if args.decimList[0] != 1:
        if  valid_total_loss < bestValid_total_loss:
            bestValid_total_loss = valid_total_loss
            res["params"] = str(args)
            res["bestValidLoss"] =bestValid_total_loss
            res["epochOnBestPrep"] = epoch
            print("new best loss, model saved")
            torch.save(net.state_dict(), args.foldName + '/' + args.modelName + '/' + args.modelName)
            earlyStop = 0

        else:
            earlyStop += 1
            print("increasing early stopping")

    totalTest = 0
    correct = 0
    correctrepeat = 0

    accuraList = [0] * int(args.lenPred/decim)
    musicalD = 0
    accurTest = 0
    accurRepeat = 0
    if earlyStop > 10:
        print("early stopping!")
        net.load_state_dict(torch.load(args.foldName + '/' + args.modelName + '/' + args.modelName, map_location = args.device))
        
        for local_batch, local_labels, local_key, local_beat in testing_generator:
            torch.cuda.empty_cache()
            if len(args.decimList) == 1:
                local_batch = local_batch[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous()
            local_labels = local_labels[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous() 
            local_batch, local_labels = local_batch.to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True) 
            if args.modelType == "mlpDecim":
                with torch.no_grad():
                    net.eval() 
                    net.zero_grad()
                    output = net(local_batch)
                    loss = criterion(output, local_labels)
                    test_total_loss += loss


            if args.decimList[0] == 1:
                for i in range(output.size()[0]):
                    repeat = local_batch[i][int(args.lenSeq/decim) - 1]
                    for j in range(int(args.lenPred/decim)):
                        totalTest += 1
                        correctrepeat += (repeat.max(0)[1] == local_labels[i][j].max(0)[1]).item()
                        result = (output[i][j].max(0)[1] == local_labels[i][j].max(0)[1]).item()
                        correct += result
                        accuraList[j] += result
                        #if args.dist != 'None':
                        	#musicalD += torch.dot(torch.matmul(output[i][j], distMat), local_labels[i][j]) #fix distMat calculation !!
        if args.decimList[0] == 1:
            accurTest = 100 * correct / totalTest 
            accurRepeat = 100 * correctrepeat / totalTest
            accuraList[:] = [x / (totalTest/(int(args.lenPred/args.decimList[0]))) for x in accuraList]
        print('Best accuracy of the network on test dataset: {} %'.format(accurTest))
        res["bestAccurTest"] = accurTest
        res["repeatAccurTest"] = accurRepeat
       # if args.dist != 'None':
            #res["musicalDistonTestWithBestValAcc"] = musicalD
        res["bestAccurTestList"] = accuraList
        sauv = open(args.foldName + '/' + args.modelName + '/' + "res" + args.modelName + ".pkl","wb")
        pickle.dump(res,sauv)
        sauv.close()
        print("End of training")
        break    
            
