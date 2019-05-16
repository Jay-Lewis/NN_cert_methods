
def run_Lip_Lin(model, inputs, targets, true_labels, true_ids, norm='2', method = 'ours',eps = 0.2, numlayer = None):

    # print("Evaluating", modelfile)
    sys.stdout.flush()

    random.seed(1215)
    np.random.seed(1215)
    tf.set_random_seed(1215)

    # the weights and bias are saved in lists: weights and bias
    # weights[i-1] gives the ith layer of weight and so on
    weights, biases = get_weights_list(model)


    # inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=args.numimage, targeted=targeted,
    #                                                                  random_and_least_likely=True, target_type=target_type,
    #                                                                  predictor=model.model.predict, start=args.startimage)

    # Set args
    warmup = False
    lipsbnd = 'disable'
    LP = False
    LPFULL = False
    lipsteps = 30
    steps = 15


    # get the logit layer predictions
    preds = model.model.predict(inputs)
    Nsamp = 0
    r_sum = 0.0
    r_gx_sum = 0.0
    targeted = True
    modelfile = 'torch_network'

    # warmup
    if warmup:
        print("warming up...")
        sys.stdout.flush()
        if method == "spectral":
            robustness_gx = spectral_bound(weights, biases, 0, 1, inputs[0], preds[0], numlayer, norm,
                                           not targeted)
        else:
            compute_worst_bound(weights, biases, 0, 1, inputs[0], preds[0], numlayer, norm, 0.01, method,
                                lipsbnd, LP, LPFULL, not targeted)
    print("starting robustness verification on {} images!".format(len(inputs)))
    sys.stdout.flush()
    sys.stderr.flush()
    total_time_start = time.time()

    min_dists = []
    times = []

    for i in range(len(inputs)):
        Nsamp += 1
        p = norm  # p = "1", "2", or "i"
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        start = time.time()
        # Spectral bound: no binary search needed
        if method == "spectral":
            robustness_gx = spectral_bound(weights, biases, predict_label, target_label, inputs[i], preds[i], numlayer,
                                           p, not targeted)
        # compute worst case bound
        # no need to pass in sess, model and data
        # just need to pass in the weights, true label, norm, x0, prediction of x0, number of layer and eps
        elif lipsbnd != "disable":
            # You can always use the "multi" version of Lipschitz bound to improve results (about 30%).
            robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label, inputs[i], preds[i],
                                                      numlayer, p, eps, lipsteps, method, lipsbnd,
                                                      not targeted)
            eps = eps
            # if initial eps is too small, then increase it
            if robustness_gx == eps:
                while robustness_gx == eps:
                    eps = eps * 2
                    print("==============================")
                    print("increase eps to {}".format(eps))
                    print("==============================")
                    robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label, inputs[i],
                                                              preds[i], numlayer, p, eps, lipsteps, method,
                                                              lipsbnd, not targeted)
                    # if initial eps is too large, then decrease it
            elif robustness_gx <= eps / 5:
                while robustness_gx <= eps / 5:
                    eps = eps / 5
                    print("==============================")
                    print("increase eps to {}".format(eps))
                    print("==============================")
                    robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label, inputs[i],
                                                              preds[i], numlayer, p, eps, lipsteps, method,
                                                              lipsbnd, not targeted)
        else:
            gap_gx = 100
            eps = eps
            eps_LB = -1
            eps_UB = 1
            counter = 0
            is_pos = True
            is_neg = True

            # perform binary search
            eps_gx_UB = np.inf
            eps_gx_LB = 0.0
            is_pos = True
            is_neg = True
            # eps = eps_gx_LB*2
            eps = eps
            while eps_gx_UB - eps_gx_LB > 0.00001:
                gap_gx, _, _ = compute_worst_bound(weights, biases, predict_label, target_label, inputs[i], preds[i],
                                                   numlayer, p, eps, method, "disable", LP, LPFULL,
                                                   not targeted)
                print("[L2][binary search] step = {}, eps = {:.5f}, gap_gx = {:.2f}".format(counter, eps, gap_gx))
                if gap_gx > 0:
                    if gap_gx < 0.01:
                        eps_gx_LB = eps
                        break
                    if is_pos:  # so far always > 0, haven't found eps_UB
                        eps_gx_LB = eps
                        eps *= 10
                    else:
                        eps_gx_LB = eps
                        eps = (eps_gx_LB + eps_gx_UB) / 2
                    is_neg = False
                else:
                    if is_neg:  # so far always < 0, haven't found eps_LB
                        eps_gx_UB = eps
                        eps /= 2
                    else:
                        eps_gx_UB = eps
                        eps = (eps_gx_LB + eps_gx_UB) / 2
                    is_pos = False
                counter += 1
                if counter >= steps:
                    break

            robustness_gx = eps_gx_LB

        r_gx_sum += robustness_gx
        print(
            "[L1] model = {}, seq = {}, id = {}, true_class = {}, target_class = {} robustness_gx = {:.5f}, avg_robustness_gx = {:.5f}, time = {:.4f}, total_time = {:.4f}".format(
                modelfile, i, true_ids[i], predict_label, target_label,  robustness_gx, r_gx_sum / Nsamp,
                time.time() - start, time.time() - total_time_start))
        times.append(time.time() - start)
        min_dists.append(robustness_gx)
        sys.stdout.flush()
        sys.stderr.flush()

    print("[L0] model = {}, avg robustness_gx = {:.5f}, numimage = {}, total_time = {:.4f}".format(modelfile,
                                                                                                   r_gx_sum / Nsamp, Nsamp,
                                                                                                   time.time() - total_time_start))
    sys.stdout.flush()
    sys.stderr.flush()

    return min_dists, times


######################################################################################################
#
#
#
#
#
######################################################################################################


"""
main.py

main interfacing function

Copyright (C) 2018, Lily Weng  <twweng@mit.edu>
                    Huan Zhang <ecezhang@ucdavis.edu>
                    Honge Chen <chenhg@mit.edu>
"""
import save_nlayer_weights as nl
import numpy as np
from tensorflow.contrib.keras.api.keras.layers import Dense
import argparse

from setup_mnist import MNIST
from setup_cifar import CIFAR

import tensorflow as tf
import os
import sys
import random
import time

from nn_utils import generate_data
from PIL import Image

# import our bounds
from get_bounds_ours import get_weights_list, compute_worst_bound, compute_worst_bound_multi
# import others bounds: currently, LP bound is called in the compute_worst_bound
from get_bounds_others import spectral_bound


#### parser ####
parser = argparse.ArgumentParser(description='compute activation bound for CIFAR and MNIST')
parser.add_argument('--model',
                    default="mnist",
                    choices=["mnist", "cifar"],
                    help='model to be used')
parser.add_argument('--eps',
                    default=0.005,
                    type=float,
                    help="epsilon for verification")
parser.add_argument('--hidden',
                    default=1024,
                    type=int,
                    help="number of hidden neurons per layer")
parser.add_argument('--numlayer',
                    default=2,
                    type=int,
                    help='number of layers in the model')
parser.add_argument('--numimage',
                    default=10,
                    type=int,
                    help='number of images to run')
parser.add_argument('--startimage',
                    default=0,
                    type=int,
                    help='start image')
parser.add_argument('--norm',
                    default="i",
                    type=str,
                    choices=["i", "1", "2"],
                    help='perturbation norm: "i": Linf, "1": L1, "2": L2')
parser.add_argument('--method',
                    default="ours",
                    type=str,
                    choices=["ours", "spectral", "naive"],
                    help='"ours": our proposed bound, "spectral": spectral norm bounds, "naive": naive bound')
parser.add_argument('--lipsbnd',
                    type=str,
                    default="disable",
                    choices=["disable", "fast", "naive", "both"],
                    help='compute Lipschitz bound, after using some method to compute neuron lower/upper bounds')
parser.add_argument('--lipsteps',
                    type=int,
                    default=30,
                    help='number of steps to use in lipschitz bound')
parser.add_argument('--LP',
                    action="store_true",
                    help='use LP to get bounds for final output')
parser.add_argument('--LPFULL',
                    action="store_true",
                    help='use FULL LP to get bounds for output')
parser.add_argument('--warmup',
                    action="store_true",
                    help='warm up before the first iteration')
parser.add_argument('--modeltype',
                    default="vanilla",
                    choices=["vanilla", "dropout", "distill", "adv_retrain"],
                    help="select model type")
parser.add_argument('--targettype',
                    default="least",
                    choices=["untargeted", "least", "top2", "random"],
                    help='untargeted minimum distortion')
parser.add_argument('--steps',
                    default=15,
                    type=int,
                    help='how many steps to binary search')

args = parser.parse_args()
nhidden = args.hidden

targeted = True
if args.targettype == "least":
    target_type = 0b0100
elif args.targettype == "top2":
    target_type = 0b0001
elif args.targettype == "random":
    target_type = 0b0010
elif args.targettype == "untargeted":
    target_type = 0b10000
    targeted = False

if args.modeltype == "vanilla":
    suffix = ""
else:
    suffix = "_" + args.modeltype

# try models/mnist_3layer_relu_1024
modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_relu_" + str(nhidden) + suffix
if not os.path.isfile(modelfile):
    # if not found, try models/mnist_3layer_relu_1024_1024
    modelfile += ("_" + str(nhidden)) * (args.numlayer - 2) + suffix
    # if still not found, try models/mnist_3layer_relu
    if not os.path.isfile(modelfile):
        modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_relu" + suffix
        # if still not found, try models/mnist_3layer_relu_1024_best
        if not os.path.isfile(modelfile):
            modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_relu_" + str(
                nhidden) + suffix + "_best"
            if not os.path.isfile(modelfile):
                raise (RuntimeError("cannot find model file"))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # if args.model == "mnist":
    #     data = MNIST()
    #     model = nl.NLayerModel([nhidden] * (args.numlayer - 1), modelfile)
    #
    # elif args.model == "cifar":
    #     data = CIFAR()
    #     model = nl.NLayerModel([nhidden] * (args.numlayer - 1), modelfile, image_size=32, image_channel=3)
    # else:
    #     raise (RuntimeError("unknown model: " + args.model))

    # print("Evaluating", modelfile)

    # ##################################################################################
    # #                                                                                #
    # #                       Modding                                                  #
    # #                                                                                #
    # ##################################################################################

    # =====================
    # Imports
    # =====================
    import sys
    import time
    import pickle
    import torch
    import mnist.mnist_loader as  ml
    import numpy as np
    from save_nlayer_weights import NLayerModel_comparison, NLayerModel
    import tensorflow as tf
    import utilities_geo as utils_geo

    ##################################################################################
    #                                                                                #
    #                       Network Model + Data Loading                             #
    #                                                                                #
    ##################################################################################

    NETWORK_NAME = '17_mnist_small.pkl'
    MNIST_DIM = 784
    layer_sizes = [MNIST_DIM, 10, 50, 10, 2]
    trainset = ml.load_single_digits('train', [1, 7], batch_size=16,
                                     shuffle=False)
    valset = ml.load_single_digits('val', [1, 7], batch_size=16,
                                   shuffle=False)

    network = pickle.load(open(NETWORK_NAME, 'rb'))
    net = network.net
    print("Loaded pretrained network")

    # =====================
    # Set Images to Verify
    # =====================
    num_batches = len(valset)
    num_batches = 1
    images = torch.cat([batch_tuple[0] for batch_tuple in valset[0:num_batches]])
    labels = torch.cat([batch_tuple[1] for batch_tuple in valset[0:num_batches]])

    # =====================
    # Convert Network
    # =====================
    params = layer_sizes[1:-1]
    nlayers = len(params)
    restore = [None]
    for fc in network.fcs:
        weight = utils_geo.as_numpy(fc.weight).T
        bias = utils_geo.as_numpy(fc.bias)
        restore.append([weight, bias])
        restore.append(None)

    #     FL_network = NLayerModel_comparison(params, restore=restore, session=sess)
    # FL_network = NLayerModel_comparison(params, session=sess)
    restore = 'models/mnist_2layer_relu_20_best'
    params = [nhidden] * (args.numlayer - 1)
    model = NLayerModel(params, restore=modelfile)

    # =====================================

    sys.stdout.flush()

    random.seed(1215)
    np.random.seed(1215)
    tf.set_random_seed(1215)

    # the weights and bias are saved in lists: weights and bias
    # weights[i-1] gives the ith layer of weight and so on
    weights, biases = get_weights_list(model)

    # inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=args.numimage, targeted=targeted,
    #                                                                  random_and_least_likely=True,
    #                                                                  target_type=target_type,
    #                                                                  predictor=model.model.predict,
    #                                                                  start=args.startimage)

    # =====================
    # Convert Images
    # =====================
    images_lin_lip = utils_geo.as_numpy(images.reshape(-1, 28, 28, 1))

    true_labels = []
    targets = []
    for label in labels:
        if label == 0:
            true_labels.append([1.0, 0.0])
            targets.append([0.0, 1.0])
        elif label == 1:
            true_labels.append([0.0, 1.0])
            targets.append([1.0, 0.0])
    true_ids = [num for num in range(0, np.shape(images)[0] + 1)]
    inputs = images_lin_lip


    # ##################################################################################
    # #                                                                                #
    # #                       Modding Done                                             #
    # #                                                                                #
    # ##################################################################################



    # get the logit layer predictions
    preds = model.model.predict(inputs)

    Nsamp = 0
    r_sum = 0.0
    r_gx_sum = 0.0

    # warmup
    if args.warmup:
        print("warming up...")
        sys.stdout.flush()
        if args.method == "spectral":
            robustness_gx = spectral_bound(weights, biases, 0, 1, inputs[0], preds[0], args.numlayer, args.norm,
                                           not targeted)
        else:
            compute_worst_bound(weights, biases, 0, 1, inputs[0], preds[0], args.numlayer, args.norm, 0.01,
                                args.method, args.lipsbnd, args.LP, args.LPFULL, not targeted)
    print("starting robustness verification on {} images!".format(len(inputs)))
    sys.stdout.flush()
    sys.stderr.flush()
    total_time_start = time.time()
    for i in range(len(inputs)):
        Nsamp += 1
        p = args.norm  # p = "1", "2", or "i"
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        start = time.time()
        # Spectral bound: no binary search needed
        if args.method == "spectral":
            robustness_gx = spectral_bound(weights, biases, predict_label, target_label, inputs[i], preds[i],
                                           args.numlayer, p, not targeted)
        # compute worst case bound
        # no need to pass in sess, model and data
        # just need to pass in the weights, true label, norm, x0, prediction of x0, number of layer and eps
        elif args.lipsbnd != "disable":
            # You can always use the "multi" version of Lipschitz bound to improve results (about 30%).
            robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label, inputs[i],
                                                      preds[i], args.numlayer, p, args.eps, args.lipsteps,
                                                      args.method, args.lipsbnd, not targeted)
            eps = args.eps
            # if initial eps is too small, then increase it
            if robustness_gx == eps:
                while robustness_gx == eps:
                    eps = eps * 2
                    print("==============================")
                    print("increase eps to {}".format(eps))
                    print("==============================")
                    robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label,
                                                              inputs[i], preds[i], args.numlayer, p, eps,
                                                              args.lipsteps, args.method, args.lipsbnd,
                                                              not targeted)
                    # if initial eps is too large, then decrease it
            elif robustness_gx <= eps / 5:
                while robustness_gx <= eps / 5:
                    eps = eps / 5
                    print("==============================")
                    print("increase eps to {}".format(eps))
                    print("==============================")
                    robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label,
                                                              inputs[i], preds[i], args.numlayer, p, eps,
                                                              args.lipsteps, args.method, args.lipsbnd,
                                                              not targeted)
        else:
            gap_gx = 100
            eps = args.eps
            eps_LB = -1
            eps_UB = 1
            counter = 0
            is_pos = True
            is_neg = True

            # perform binary search
            eps_gx_UB = np.inf
            eps_gx_LB = 0.0
            is_pos = True
            is_neg = True
            # eps = eps_gx_LB*2
            eps = args.eps
            while eps_gx_UB - eps_gx_LB > 0.00001:
                gap_gx, _, _ = compute_worst_bound(weights, biases, predict_label, target_label, inputs[i],
                                                   preds[i], args.numlayer, p, eps, args.method, "disable", args.LP,
                                                   args.LPFULL, not targeted)
                print("[L2][binary search] step = {}, eps = {:.5f}, gap_gx = {:.2f}".format(counter, eps, gap_gx))
                if gap_gx > 0:
                    if gap_gx < 0.01:
                        eps_gx_LB = eps
                        break
                    if is_pos:  # so far always > 0, haven't found eps_UB
                        eps_gx_LB = eps
                        eps *= 10
                    else:
                        eps_gx_LB = eps
                        eps = (eps_gx_LB + eps_gx_UB) / 2
                    is_neg = False
                else:
                    if is_neg:  # so far always < 0, haven't found eps_LB
                        eps_gx_UB = eps
                        eps /= 2
                    else:
                        eps_gx_UB = eps
                        eps = (eps_gx_LB + eps_gx_UB) / 2
                    is_pos = False
                counter += 1
                if counter >= args.steps:
                    break

            robustness_gx = eps_gx_LB

        r_gx_sum += robustness_gx
        print(
            "[L1] model = {}, seq = {}, id = {}, true_class = {}, target_class = {}, robustness_gx = {:.5f}, avg_robustness_gx = {:.5f}, time = {:.4f}, total_time = {:.4f}".format(
                modelfile, i, true_ids[i], predict_label, target_label, robustness_gx,
                r_gx_sum / Nsamp, time.time() - start, time.time() - total_time_start))
        sys.stdout.flush()
        sys.stderr.flush()

    print("[L0] model = {}, avg robustness_gx = {:.5f}, numimage = {}, total_time = {:.4f}".format(modelfile,
                                                                                                   r_gx_sum / Nsamp,
                                                                                                   Nsamp,
                                                                                                   time.time() - total_time_start))
    sys.stdout.flush()
    sys.stderr.flush()

