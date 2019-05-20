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
sys.path.append('CertifiedReLURobustness')
sys.path.append('geometric-certificates')
import geocert_oop as geo_oop
import geocert_multi as geo
import utilities as utils
import os
import time
import pickle
import matplotlib.pyplot as plt
import torch
from os import listdir
from os.path import isfile, join
import mnist.mnist_loader as  ml
MNIST_DIM = 784


##################################################################################
#                                                                                #
#                            Comparison Loop                                     #
#                                                                                #
##################################################################################

num_img_batches = 1
batch_size = 16

# =====================
# Load Network
# =====================


model_path = os.getcwd() + "/Models/"
network_names = [f for f in listdir(model_path) if isfile(join(model_path, f))]
#TODO: change to all networks
network_names = network_names[0:2]

for network_name in network_names:

    ##################################################################################
    #                                                                                #
    #                                 Load Network                                   #
    #                                                                                #
    ##################################################################################


    filepath = model_path + network_name
    network = pickle.load(open(filepath, 'rb'))
    network.net = network.net.cpu()     # bring from gpu to cpu

    print('Network_name:', network_name)
    print('Layer_sizes:', network.layer_sizes)


    ##################################################################################
    #                                                                                #
    #                   Parameter Setting + Data Loading                             #
    #                                                                                #
    ##################################################################################

    lp_norm = 'l_2'
    if network.layer_sizes[-1] == 10:
        ALL_DIGITS = True
    else:
        ALL_DIGITS = False

    if not ALL_DIGITS:
        digits = [1, 7]
        trainset = ml.load_single_digits('train', digits, batch_size=batch_size,
                                         shuffle=False)
        valset = ml.load_single_digits('val', digits, batch_size=batch_size,
                                       shuffle=False)
        if lp_norm == 'l_inf':
            pgd_lr = 0.01
        elif lp_norm == 'l_2':
            pgd_lr = 0.005
            pgd_lr = 0.01
        num_pgd_iterations = 300

    else:
        trainset = ml.load_mnist_data('train', batch_size=batch_size, shuffle=False)
        valset = ml.load_mnist_data('val', batch_size=batch_size, shuffle=False)

        num_pgd_iterations = 300
        if lp_norm == 'l_inf':
            pgd_lr = 0.01
        elif lp_norm == 'l_2':
            pgd_lr = 0.005
        num_epochs = 100

    # =====================
    # Set Images to Verify
    # =====================
    #TODO: make data loading better
    if not ALL_DIGITS:
        images = torch.cat([batch_tuple[0] for batch_tuple in valset[0:num_img_batches]])
        labels = torch.cat([batch_tuple[1] for batch_tuple in valset[0:num_img_batches]])
    else:
        images = torch.cat([next(iter(valset))[0] for _ in range(0, num_img_batches)])
        labels = torch.cat([next(iter(valset))[1] for _ in range(0, num_img_batches)])

    print('Num Images:', len(images))



    # ##################################################################################
    # #                                                                                #
    # #                       Geocert                                                  #
    # #                                                                                #
    # ##################################################################################
    #
    # print('-------------------------------------------')
    # print('Starting Geocert')
    # print('-------------------------------------------')
    #
    # # Run Geocert on a set of mnist digits
    # min_dists = []
    # pgd_dists = []
    # num_polys = []
    # poly_maps = []
    # times = []
    # for image in images:
    #         # Builds an object used to to hold algorithm parameters
    #         cert_obj = geo.IncrementalGeoCert(network, verbose=True, config_fxn='parallel',
    #                                   config_fxn_kwargs={'num_jobs': 1},
    #                                   hyperbox_bounds=[0.0, 1.0])
    #
    #         # plt.gray()
    #         # plt.imshow(image.squeeze())
    #         # plt.show()
    #         true_label = network(image).squeeze().max(0)[1].item()
    #         # Run Geocert
    #         start = time.time()
    #         output = cert_obj.min_dist(image.view(1, -1), lp_norm=lp_norm, compute_upper_bound=
    #                                    {"optimizer_kwargs": {"lr": pgd_lr}, "num_iterations": num_pgd_iterations})
    #         lp_dist, adv_ex_bound, adv_ex, best_example, boundary_facet, seen_poly_map = output
    #         end = time.time()
    #         # plt.gray()
    #         # best_example_np = utils.as_numpy(best_example).reshape(28,28)
    #         # plt.imshow(best_example_np.squeeze())
    #         # plt.show()
    #
    #         min_dists.append(lp_dist)
    #         pgd_dists.append(adv_ex_bound)
    #         print('TIME', end-start)
    #         times.append(end-start)
    #         num_polys.append(len(seen_poly_map))
    #         poly_maps.append(seen_poly_map)
    #         print('===========================================')


    ##################################################################################
    #                                                                                #
    #                       Geocert                                                  #
    #                                                                                #
    ##################################################################################

    print('-------------------------------------------')
    print('Starting Geocert')
    print('-------------------------------------------')

    # Run Geocert on a set of mnist digits
    min_dists = []
    pgd_dists = []
    num_polys = []
    poly_maps = []
    times = []
    for i, image in enumerate(images):
            if i < 3:
                continue
            # Builds an object used to to hold algorithm parameters
            cert_obj = geo.IncrementalGeoCertMultiProc(network, verbose=True,
                                      hyperbox_bounds=[0.0, 1.0])

            true_label = network(image).squeeze().max(0)[1].item()
            # Run Geocert
            start = time.time()
            output = cert_obj.min_dist_multiproc(image.view(1, -1), num_proc=1, lp_norm=lp_norm, compute_upper_bound=
                                       {"optimizer_kwargs": {"lr": pgd_lr}, "num_iterations": num_pgd_iterations})

            lp_dist, adv_ex_bound, adv_ex, best_example, ub_time, seen_polytopes, _, _,lower_bound_times, upper_bound_times = output

            end = time.time()

            min_dists.append(lp_dist)
            pgd_dists.append(adv_ex_bound)
            print('TIME', end-start)
            times.append(end-start)
            num_polys.append(len(seen_polytopes))
            poly_maps.append(seen_polytopes)
            print('===========================================')

    # =====================
    # Save Output
    # =====================

    output_dictionary = {'min_dists': min_dists , 'pgd_dists': pgd_dists, 'num_polys': num_polys,
                        'poly_maps': [list(_) for _ in poly_maps], 'times': times}

    cwd = os.getcwd()
    filename = cwd + "/Results/Geocert_out_"+network_name[0:-4]+"_"+lp_norm+".pkl"
    f = open(filename, 'wb')
    pickle.dump(output_dictionary, f)
    f.close()



    ##################################################################################
    #                                                                                #
    #                       Fast Lin / Fast Lip                                      #
    #                                                                                #
    ##################################################################################

    print('-------------------------------------------')
    print('Starting Fast Lin / Fast Lip')
    print('-------------------------------------------')

    # =====================
    # Run Fast Lin / Fast LIP
    # =====================
    from CertifiedReLURobustness.Lip_Lin_verify import Lin_Lip_verify_modded

    dists, times = Lin_Lip_verify_modded(network, network_name, images, labels)

    ##################################################################################
    #                                                                                #
    #                       Print Comparison Summary                                 #
    #                                                                                #
    ##################################################################################


    print('===========================================')
    print('NETWORK COMPARISON SUMMARY:')
    print('===========================================')


    def mean(elements):
        return sum(elements) / float(len(elements))

    def percentile(elements, perc=50):
        perc_idx = int(len(elements) * perc / 100.0)
        return sorted(elements)[perc_idx]


    methods = ['Geocert', 'Lip_Lin']

    for method in methods:
        # =====================
        # Load and Display Output
        # =====================
        print('------------------------------------')
        print('Method:  ', method)
        print('------------------------------------')
        cwd = os.getcwd()
        filename = cwd + "/Results/" + method + "_out_"+str(network_name[0:-4])+"_"+lp_norm+".pkl"
        f = open(filename,"rb")
        output_dict = pickle.load(f)
        f.close()


        for k, v in output_dict.items():
            if k not in ['min_dists', 'pgd_dists', 'num_polys', 'times']:
                continue
            print('-' * 20, k, '-' * 20)
            print('MEAN   ', mean(v))
            print('25th   ', percentile(v, 25))
            print('MEDIAN ', percentile(v, 50))
            print('90     ', percentile(v, 90))
            print('\n')


##################################################################################
#                                                                                #
#                            Comparison Loop  END                                #
#                                                                                #
##################################################################################



