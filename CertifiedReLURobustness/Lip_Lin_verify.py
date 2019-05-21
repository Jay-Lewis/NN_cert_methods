"""
main interfacing function

Copyright (C) 2018, Lily Weng  <twweng@mit.edu>
                    Huan Zhang <ecezhang@ucdavis.edu>
                    Honge Chen <chenhg@mit.edu>
"""

import os
import random
import sys
import time
import pickle
import torch

sys.path.append('../mister_ed')
import mnist.mnist_loader as  ml
import numpy as np
from save_nlayer_weights import NLayerModel_comparison, NLayerModel
import tensorflow as tf
from utilities import as_numpy

# import our bounds
from get_bounds_ours import get_weights_list, compute_worst_bound, compute_worst_bound_multi
# import others bounds: currently, LP bound is called in the compute_worst_bound
from get_bounds_others import spectral_bound


def Lin_Lip_verify_modded(network, network_name, images, labels, norm="2", warmup="True", method="ours", targettype="untargeted",
                   lipsbnd='disable', LP=False, LPFULL=False, eps=0.2, lipsteps=30, steps=15):
    """

    :param network: "PLNN network object"
    :param images: "images to verify in format [N, 1, n1, n2] (for MNIST n1=n2=28)"
    :param labels: "labels of images. list of ints"
    :param norm: {"i", "1", "2"}
    :param warmup: {True, False}
           warm up before the first iteration
    :param method: {"ours", "spectral", "naive"}
           "ours": our proposed bound, "spectral": spectral norm bounds, "naive": naive bound'
    :param targettype: {"untargeted", "top2"}
           "tops2: 2nd highest prediction label"
    :param lipsbnd: {"disable", "fast", "naive", "both"}
           compute Lipschitz bound, after using some method to compute neuron lower/upper bounds
    :param LP: {True, False}
           use LP to get bounds for final output
    :param LPFULL: {True, False}
           use FULL LP to get bounds for output
    :param eps: "initial guess for epsilon for verification"
    :param lipsteps: "number of steps to use in lipschitz bound'
    :param steps: "how many steps to binary search"
    :return:
    """

    #TODO: eps?

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    with tf.Session() as sess:
        sys.stdout.flush()
        random.seed(1215)
        np.random.seed(1215)
        tf.set_random_seed(1215)


        ##################################################################################
        #                                                                                #
        #                       Network Model + Results Loading                             #
        #                                                                                #
        ##################################################################################

        # =====================
        # Convert Network
        # =====================
        layer_sizes = network.layer_sizes
        num_classes = layer_sizes[-1]
        params = layer_sizes[1:-1]
        restore = [None]
        for fc in network.fcs:
            weight =  as_numpy(fc.weight).T
            bias =  as_numpy(fc.bias)
            restore.append([weight, bias])
            restore.append(None)

        model = NLayerModel_comparison(params, num_classes=num_classes, restore=restore)
        # FL_network = NLayerModel_comparison(params, session=sess)
        numlayer = len(params) + 1
        # restore = 'models/mnist_2layer_relu_20_best'
        # params = [nhidden] * (numlayer - 1)
        # model = NLayerModel(params, restore=modelfile)

        # =======================================
        # Convert Image Shapes + Create targets
        # =======================================
        images_lin_lip =  as_numpy(images.reshape(-1, 28, 28, 1))

        true_labels = []
        top2_targets = []
        preds = []
        for image, label in zip(images, labels):
            label_array = np.zeros([num_classes])
            label_array[label] = 1
            true_labels.append(label_array)
            pred = model.predict(tf.constant( as_numpy(image)))
            array = pred.eval()[0]
            preds.append(array)
            target_index = np.argsort(pred.eval()[0])[-2]
            top2_target = np.zeros([num_classes])
            top2_target[target_index] = 1
            top2_targets.append(top2_target)

        true_labels = np.asarray(true_labels)
        true_ids = [num for num in range(0, np.shape(images)[0] + 1)]
        inputs = images_lin_lip

        if targettype == "untargeted":
            targets = true_labels
            targeted = False
        elif targettype == "top2":
            targets = np.asarray(top2_targets)
            targeted = True
        else:
            raise NotImplementedError

        # =====================================

        sys.stdout.flush()

        random.seed(1215)
        np.random.seed(1215)
        tf.set_random_seed(1215)

        # the weights and bias are saved in lists: weights and bias
        # weights[i-1] gives the ith layer of weight and so on
        weights, biases = get_weights_list(model)
        preds = model.model.predict(inputs)

        Nsamp = 0
        r_sum = 0.0
        r_gx_sum = 0.0

        # warmup
        if warmup:
            print("warming up...")
            sys.stdout.flush()
            if method == "spectral":
                robustness_gx = spectral_bound(weights, biases, 0, 1, inputs[0], preds[0], numlayer, norm, not targeted)
            else:
                compute_worst_bound(weights, biases, 0, 1, inputs[0], preds[0], numlayer, norm, 0.01, method, lipsbnd,
                                    LP, LPFULL, not targeted)
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
                robustness_gx = spectral_bound(weights, biases, predict_label, target_label, inputs[i], preds[i],
                                               numlayer, p, not targeted)
            # compute worst case bound
            # no need to pass in sess, model and data
            # just need to pass in the weights, true label, norm, x0, prediction of x0, number of layer and eps
            elif lipsbnd != "disable":
                # You can always use the "multi" version of Lipschitz bound to improve results (about 30%).
                robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label, inputs[i],
                                                          preds[i], numlayer, p, eps, lipsteps, method, lipsbnd,
                                                          not targeted)
                eps = eps
                # if initial eps is too small, then increase it
                if robustness_gx == eps:
                    while robustness_gx == eps:
                        eps = eps * 2
                        print("==============================")
                        print("increase eps to {}".format(eps))
                        print("==============================")
                        robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label,
                                                                  inputs[i], preds[i], numlayer, p, eps, lipsteps,
                                                                  method, lipsbnd, not targeted)
                        # if initial eps is too large, then decrease it
                elif robustness_gx <= eps / 5:
                    while robustness_gx <= eps / 5:
                        eps = eps / 5
                        print("==============================")
                        print("increase eps to {}".format(eps))
                        print("==============================")
                        robustness_gx = compute_worst_bound_multi(weights, biases, predict_label, target_label,
                                                                  inputs[i], preds[i], numlayer, p, eps, lipsteps,
                                                                  method, lipsbnd, not targeted)
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
                    gap_gx, _, _ = compute_worst_bound(weights, biases, predict_label, target_label, inputs[i],
                                                       preds[i], numlayer, p, eps, method, "disable", LP, LPFULL,
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
                "[L1] seq = {}, id = {}, true_class = {}, target_class = {} robustness_gx = {:.5f}, avg_robustness_gx = {:.5f},"
                " time = {:.4f}, total_time = {:.4f}".format(i, true_ids[i], predict_label, target_label,
                                                             robustness_gx, r_gx_sum / Nsamp, time.time() - start,
                                                             time.time() - total_time_start))

            times.append(time.time() - start)
            min_dists.append(robustness_gx)

            sys.stdout.flush()
            sys.stderr.flush()

        # =====================
        # Save Output
        # =====================

        output_dictionary = {'min_dists': min_dists, 'times': times}

        cwd = os.getcwd()
        import os.path as path
        norm_text = get_lp_text(norm)
        filename = cwd + "/Results/Lip_Lin_out_" + str(network_name[0:-4]) + '_' + norm_text + ".pkl"
        f = open(filename, 'wb')
        pickle.dump(output_dictionary, f)
        f.close()
        print('Saved Results @:')
        print(filename)

        print("[L0] avg robustness_gx = {:.5f}, numimage = {}, total_time = {:.4f}".format(r_gx_sum / Nsamp, Nsamp, time.time() - total_time_start))
        sys.stdout.flush()
        sys.stderr.flush()

    sess.close()


    return min_dists, times


def get_lp_text(norm_lip):
    if norm_lip == '2':
        return 'l_2'
    elif norm_lip == 'i':
        return 'l_inf'
    else:
        raise NotImplementedError