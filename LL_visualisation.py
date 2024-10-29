# Visualising the loss landscape using techniques from https://github.com/tomgoldstein/loss-landscape/. 

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm
import itertools
import copy
import os
import re
from models.resnet import ResNet18_cifar, ResNet18_mnist
from models.vgg import VGG_cifar, VGG_mnist

def get_dataloaders(batch_size, dataset):
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])        
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    elif dataset == 'fashionmnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader


def train(trainloader, net, criterion, optimizer):
    net.train()
    train_loss = 0
    total = 0
    for _, (inputs, targets) in tqdm(enumerate(trainloader)):
        batch_size = inputs.size(0)
        total += batch_size
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*batch_size
    return train_loss/total

def test(testloader, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in tqdm(enumerate(testloader)):
            batch_size = inputs.size(0)
            total += batch_size
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()
    return test_loss/total, 100.*correct/total

def name_save_folder(args):
    save_folder = args.model + '_' + str(args.optimizer)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_dataset=' + str(args.dataset)
    save_folder += '_epochs=' + str(args.epochs)
    return save_folder

def filternorm(d, w):
    for dd, ww in zip(d, w):
        dd.mul_(ww.norm()/(dd.norm() + 1e-10))
    return d

def random_direction(net):
    weights = [p.data for p in net.parameters()]
    direction = [torch.randn(w.size()) for w in weights]
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            # Ignore batch normalisation 
            d.fill_(0)
        else:
            # Normalise directions corresponding to weights/biases
            filternorm(d, w)
    return direction

def linear(net, alpha, beta, theta1, directions):
    "Regular linear interpolation"
    dx, dy = directions
    changes = [d0*alpha + d1*beta for (d0, d1) in zip(dx, dy)]
    for (p, w, d) in zip(net.parameters(), theta1, changes):
        p.data = w + torch.Tensor(d).type(type(w))
        
def landscape(net, criterion, args, interpolate):
    if args.train_mode == 'y':
        net.train()
    elif args.train_mode == 'n':
        net.eval()
    initial_net = copy.deepcopy(net)
    theta1 = [p.data for p in initial_net.parameters()]
    xmin, xmax, xnum = [float(a) for a in args.x.split(':')]
    ymin, ymax, ynum = [float(a) for a in args.y.split(':')]
    xnum = int(xnum); ynum = int(ynum)
    x = np.linspace(xmin, xmax, xnum)
    y = np.linspace(ymin, ymax, ynum)
    L = np.zeros((xnum, ynum))
    # Create axes directions
    xdirection = random_direction(net); ydirection = random_direction(net)
    # Calculate loss
    inputs, targets = next(iter(trainloader))
    if args.shuffle == 'y': # Shuffle/randomise the labels 
        targets = torch.randint(0, 10, targets.shape)
    with torch.no_grad():
        for (i, alpha), (j, beta) in tqdm(itertools.product(enumerate(x), enumerate(y))):
            interpolate(net, alpha, beta, theta1, [xdirection, ydirection])
            if i == 0 and j == 0:
                assert net != initial_net, "Bug: networks are the same"
            outputs = net(inputs)
            L[i,j] = criterion(outputs, targets).item()
    return L   

def main(args=None):
    # Training options
    parser = argparse.ArgumentParser(description='Visualising the loss landscape for training on FashionMNIST')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--optimizer', default='adam', help='optimizer: sgd | adam')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save_every', default=1, type=int, help='save every n epochs')
    parser.add_argument('--dataset', default='fashionmnist', type=str, help='dataset: cifar10 | fashionmnist')
    # # model parameters
    parser.add_argument('--model', '-m', default='vgg', help='model: vgg | resnet')
    parser.add_argument('--load_model', type=str, help='path to pretrained model')
    # visualisation parameters
    parser.add_argument('--visualize', default='n', type=str, help='whether to visualise the loss landscape: y | n')
    parser.add_argument('--interpolate', default='linear', help='interpolation method: linear')
    parser.add_argument('--x', default='-2.4:2.4:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default='-2.4:2.4:51', help='A string with format ymin:ymax:ynum')
    parser.add_argument('--train_mode', default='y', help='visualise in train or eval mode: y | n')
    parser.add_argument('--shuffle', default='n', help='shuffle labels: y | n')
    # To allow running through notebook with ipython
    parser.add_argument('--f')

    args = parser.parse_args()
    
    # Set the seed for reproducing the results
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # Get dataloaders
    trainloader, testloader = get_dataloaders(args.batch_size, args.dataset)
    
    # Create model
    if args.model == 'vgg':
        if args.dataset == 'fashionmnist':
            net = VGG_mnist()
        elif args.dataset == 'cifar10':
            net = VGG_cifar()
    elif args.model == 'resnet':
        if args.dataset == 'fashionmnist':
            net = ResNet18_mnist()
        elif args.dataset == 'cifar10':
            net = ResNet18_cifar()
    print(net)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Create save folder if not loading a model
    if args.load_model is None:
        save_folder = name_save_folder(args)
        if not os.path.exists('trained_nets/' + save_folder):
            os.makedirs('trained_nets/' + save_folder)  
    
    if args.load_model is not None:
        print("Loading pretrained model")
        state = torch.load(args.load_model)
        net.load_state_dict(state['state_dict'])
    else:
        print("Training from scratch")
        # Optimizer; customisation of hyperparameters to be added
        if args.optimizer == 'adam':
            optimizer = optim.Adam(net.parameters()) # ADAM with default LR = 0.001, betas=(0.9, 0.999)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=0.001) # SGD with default LR = 0.001

        # Training; default is 1 epoch
        for epoch in range(args.epochs):
            train_loss = train(trainloader, net, criterion, optimizer)
            print(f"Epoch {epoch+1} Loss: {train_loss}")
            if (epoch % args.save_every == 0) and (epoch != args.epochs - 1):
                test_loss, test_acc = test(testloader, net, criterion)
                print("Test loss: ", test_loss, " Test accuracy: ", test_acc)
                # Save the model
                state = {
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                }
                opt_state = {
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, 'trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
                torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7')
        
        test_loss, test_acc = test(testloader, net, criterion)
        print("Test loss: ", test_loss, " Test accuracy: ", test_acc)
            
        # Save the model
        state = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'epoch': epoch,
            'state_dict': net.state_dict(),
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, 'trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
        torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7')
    
    if args.visualize == 'y':
        # Visualise the loss landscape
        if args.interpolate == 'linear':
            interpolate = linear
        # Whether to visualise in train or eval mode
        if args.train_mode == 'y':
            train_or_eval = 'train'
        elif args.train_mode == 'n':
            train_or_eval = 'eval'
        if args.shuffle == 'y':
            shuffle = 'shuffle'
        else:
            shuffle = ''
        L = landscape(net, criterion, args, interpolate)
        if args.load_model is not None:
            head, tail = os.path.split(args.load_model)
            regex = re.compile(r'\d+')
            nums_in_tail = regex.findall(tail)
            if nums_in_tail == []:
                epoch = 0
            else:
                epoch = int(nums_in_tail[0])
            np.save(head + '/loss_landscape_' + str(epoch) + '_' + str(args.interpolate) + '_' + train_or_eval + '_' + shuffle + '.npy', L)
        else:
            np.save('trained_nets/' + save_folder + '/loss_landscape_' + str(epoch) + + '_' + str(args.interpolate) + '_' + train_or_eval + '_' + shuffle + '.npy', L)

if __name__ == '__main__':
    main()