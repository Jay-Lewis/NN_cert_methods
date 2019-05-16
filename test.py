##################################################################################
#                                                                                #
#                                Comparison Notebook                             #
#                                                                                #
##################################################################################

# =====================
# Imports
# =====================
# %load_ext line_profiler
import sys
sys.path.append('..')
sys.path.append('mister_ed') # library for adversarial examples
sys.path.append('Geocert')
from collections import defaultdict
import geocert_oop as geo
from domains import Domain
from plnn import PLNN
import _polytope_ as _poly_
from _polytope_ import Polytope, Face
import utilities as utils
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from cvxopt import solvers, matrix
import adversarial_perturbations as ap
import prebuilt_loss_functions as plf
import loss_functions as lf
import adversarial_attacks as aa
import utils.pytorch_utils as me_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import mnist.mnist_loader as  ml
MNIST_DIM = 784


##################################################################################
#                                                                                #
#                       Network Model + Data Loading                             #
#                                                                                #
##################################################################################

# Define functions to train and evaluate a network

def l1_loss(net):
    return sum([_.norm(p=1) for _ in net.parameters() if _.dim() > 1])


def l2_loss(net):
    return sum([_.norm(p=2) for _ in net.parameters() if _.dim() > 1])


def train(net, trainset, num_epochs):
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    for epoch in range(num_epochs):
        err_acc = 0
        err_count = 0
        for data, labels in trainset:
            output = net(Variable(data.view(-1, 784)))
            l = nn.CrossEntropyLoss()(output, Variable(labels)).view([1])
            l1_scale = torch.Tensor([2e-3])
            l += l1_scale * l1_loss(net).view([1])

            err_acc += (output.max(1)[1].data != labels).float().mean()
            err_count += 1
            opt.zero_grad()
            (l).backward()
            opt.step()
        print("(%02d) error:" % epoch, err_acc / err_count)


def test_acc(net, valset):
    err_acc = 0
    err_count = 0
    for data, labels in valset:
        n = data.shape[0]
        output = net(Variable(data.view(-1, 784)))
        err_acc += (output.max(1)[1].data != labels).float().mean() * n
        err_count += n

    print("Accuracy of: %.03f" % (1 - (err_acc / err_count).item()))


NETWORK_NAME = '17_mnist_small.pkl'
ONE_SEVEN_ONLY = True
layer_sizes = [MNIST_DIM, 10, 50, 10, 2]
if ONE_SEVEN_ONLY:
    trainset = ml.load_single_digits('train', [1, 7], batch_size=16,
                                     shuffle=False)
    valset = ml.load_single_digits('val', [1, 7], batch_size=16,
                                   shuffle=False)
else:
    trainset = ml.load_mnist_data('train', batch_size=128, shuffle=False)
    valset = ml.load_mnist_data('val', batch_size=128, shuffle=False)

try:
    network = pickle.load(open(NETWORK_NAME, 'rb'))
    net = network.net
    print("Loaded pretrained network")
except:
    print("Training a new network")

    network = PLNN(layer_sizes)
    net = network.net
    train(net, trainset, 10)
    pickle.dump(network, open(NETWORK_NAME, 'wb'))

test_acc(net, valset)


# =====================
# Set Images to Verify
# =====================
num_batches = len(valset)
num_batches = 10
images = torch.cat([batch_tuple[0] for batch_tuple in valset[0:num_batches]])



# =====================
# Imports
# =====================
import numpy as np
from CertifiedReLURobustness.save_nlayer_weights import NLayerModel_comparison, NLayerModel
import tensorflow as tf

# ##################################################################################
# #                                                                                #
# #                       Fast-Lin / Fast-Lip                                      #
# #                                                                                #
# ##################################################################################

# =====================
# Convert Network
# =====================
# params = layer_sizes[1:-1]
# restore = [None, ]
# for fc in network.fcs:
#     weight = utils.as_numpy(fc.weight).T
#     bias = utils.as_numpy(fc.bias)
#     restore.append([weight, bias])
#     restore.append(None)

params = layer_sizes[1:-1]
nlayers = len(params)
restore = [None]
for fc in network.fcs:
    weight = utils.as_numpy(fc.weight).T
    bias = utils.as_numpy(fc.bias)
    restore.append([weight, bias])
    restore.append(None)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# with tf.Session() as sess:
#

FL_network = NLayerModel_comparison(params, restore=restore)



# # =====================
# # Save Network
# # =====================
#
#
# import tensorflow as tf
# import scipy.io as sio
# with tf.Session() as sess:
#
#     print(FL_network.W.get_weights())
#     [W, bias_W] = FL_network.W.get_weights()
#     save_dict = {'W': W, 'bias_W': bias_W}
#     print("Output layer shape:", W.shape)
#     U = FL_network.U
#     for i, Ui in enumerate(U):
#         # save hidden layer weights, layer by layer
#         [weight_Ui, bias_Ui] = Ui.get_weights()
#         print("Hidden layer {} shape: {}".format(i, weight_Ui.shape))
#         save_dict['U'+str(i+1)] = weight_Ui
#         save_dict['bias_U'+str(i+1)] = bias_Ui
#
#     save_name = 'torch' + "_" + str(nlayers) + "layers"
#     print('saving to {}.mat with matrices {}'.format(save_name, save_dict.keys()))
#     print(save_name)
#     # results saved to mnist.mat or cifar.mat
#     sio.savemat(save_name, save_dict)


# # =====================
# # Test Load Network
# # =====================
# import scipy.io as sio
#
# save_name = 'torch' + "_" + str(nlayers) + "layers"
# loaded_weights = sio.loadmat(save_name)
#
# test_model = NLayerModel(loaded_weights)
#
# for image in images[0:5]:
#     image = utils.as_numpy(image).reshape(1, 28, 28, 1)
#     print(FL_network.model.predict(image))
# print('-------')
# for image in images[0:5]:
#     image = utils.as_numpy(image).reshape(1, 28, 28, 1)
#     print(test_model.model.predict(image))