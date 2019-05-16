#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_bound_ours.py

core functions for Fast-Lin and Fast-Lip bounds

Copyright (C) 2018, Lily Weng  <twweng@mit.edu>
                    Huan Zhang <ecezhang@ucdavis.edu>
                    Honge Chen <chenhg@mit.edu>
"""


from numba import jit
import numpy as np
from get_bounds_others import get_layer_bound_LP

# use dictionary to save weights and bias
# use list to save "transposed" weights and bias
# e.g. for a 2 layer network with nodes 784 (input), 1024 (hidden), 10
# after transposed, shape of weights[0] = 1024*784, weights[1] = 10*1024 
def get_weights_list(model):
    
    weights = []
    bias = []
    
    U = model.U    
    for i, Ui in enumerate(U):
        # save hidden layer weights, layer by layer
        # middle layer weights: Ui
        [weight_Ui, bias_Ui] = Ui.get_weights()
        print("Hidden layer {} weight shape: {}".format(i, weight_Ui.shape))        
        weights.append(np.ascontiguousarray(np.transpose(weight_Ui)))
        bias.append(np.ascontiguousarray(np.transpose(bias_Ui)))
        print("Hidden layer {} bias shape: {}".format(i,bias_Ui.shape))

    # last layer weights: W
    [W, bias_W] = model.W.get_weights()    
    weights.append(np.ascontiguousarray(np.transpose(W)))
    bias.append(np.ascontiguousarray(np.transpose(bias_W)))
    print("Last layer weight shape: {}".format(W.shape))
    print("Last layer bias shape: {}".format(bias_W.shape))
    
    return weights, bias   

@jit(nopython=True)
def ReLU(vec):
    
    return np.maximum(vec, 0)

@jit(nopython=True)
def get_layer_bound(W_Nk,b_Nk,UB_prev,LB_prev,is_last,x0,eps,p_n):

    gamma = np.empty_like(W_Nk)
    # gamma = np.transpose(gamma)
    eta = np.empty_like(gamma)
    
    UB_Nk = np.empty_like(b_Nk)
    LB_Nk = np.empty_like(b_Nk)
    
    UB_new = np.empty_like(b_Nk)
    LB_new = np.empty_like(b_Nk)

    #print("W_Nk shape")
    #print(W_Nk.shape)
    
    # I reordered the indices for faster sequential access, so gamma and eta are now transposed
    for ii in range(W_Nk.shape[0]):
        for jj in range(W_Nk.shape[1]):
            if W_Nk[ii,jj] > 0:
                gamma[ii,jj] = UB_prev[jj]
                eta[ii,jj] = LB_prev[jj]
            else:
                gamma[ii,jj] = LB_prev[jj]
                eta[ii,jj] = UB_prev[jj]
              
        UB_Nk[ii] = np.dot(W_Nk[ii], gamma[ii])+b_Nk[ii]
        LB_Nk[ii] = np.dot(W_Nk[ii], eta[ii])+b_Nk[ii]
        #print('UB_Nk[{}] = {}'.format(ii,UB_Nk[ii]))
        #print('LB_Nk[{}] = {}'.format(ii,LB_Nk[ii]))
    
    Ax0 = np.dot(W_Nk,x0)
    for j in range(W_Nk.shape[0]):

        if p_n == 105: # p == "i", q = 1
            dualnorm_Aj = np.sum(np.abs(W_Nk[j]))
        elif p_n == 1: # p = 1, q = i
            dualnorm_Aj = np.max(np.abs(W_Nk[j]))
        elif p_n == 2: # p = 2, q = 2
            dualnorm_Aj = np.linalg.norm(W_Nk[j])
        
        UB_new[j] = Ax0[j]+eps*dualnorm_Aj+b_Nk[j]
        LB_new[j] = Ax0[j]-eps*dualnorm_Aj+b_Nk[j]
        
    is_old = False

    if is_last: # the last layer has no ReLU
        if is_old:
            return UB_Nk, LB_Nk
        else:
            return UB_new, LB_new
    else:# middle layers
        return ReLU(UB_Nk), ReLU(LB_Nk)

# bound a list of A matrix
def init_layer_bound_relax_matrix_huan(Ws):
    nlayer = len(Ws)
    # preallocate all A matrices
    diags = [None] * nlayer
    # diags[0] will be skipped
    diags[0] = np.ones(1, dtype=np.float32)
    for i in range(1,nlayer):
        diags[i] = np.empty(Ws[i].shape[1], dtype=np.float32)
    return diags

# matrix version of get_layer_bound_relax
@jit(nopython=True)
def get_layer_bound_relax_matrix_huan_optimized(Ws,bs,UBs,LBs,neuron_state,nlayer,diags,x0,eps,p_n):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1) == len(diags)

    # step 1: create auxillary arrays; we have only nlayer-1 layers of activations
    # we only need to create for this new layer
    idx_unsure = np.nonzero(neuron_state[nlayer - 2] == 0)[0]

    # step 2: calculate all D matrices, there are nlayer such matrices
    # only need to create diags for this layer
    alpha = neuron_state[nlayer - 2].astype(np.float32)
    np.maximum(alpha, 0, alpha)
    alpha[idx_unsure] = UBs[nlayer-1][idx_unsure]/(UBs[nlayer-1][idx_unsure] - LBs[nlayer-1][idx_unsure])
    diags[nlayer-1][:] = alpha

    # step 3: update matrix A (merged into one loop)
    # step 4: adding all constants (merged into one loop)
    constants = np.copy(bs[-1]) # the last bias

    # step 5: bounding l_n term for each layer
    UB_final = np.zeros_like(constants)
    LB_final = np.zeros_like(constants)
    # first A is W_{nlayer} D_{nlayer}
    A = Ws[nlayer-1] * diags[nlayer-1]
    for i in range(nlayer-1, 0, -1):
        # constants of previous layers
        constants += np.dot(A, bs[i-1])
        # unsure neurons of this layer
        idx_unsure = np.nonzero(neuron_state[i-1] == 0)[0]
        # create l array for this layer
        l_ub = np.empty_like(LBs[i])
        l_lb = np.empty_like(LBs[i])
        # bound the term A[i] * l_[i], for each element
        for j in range(A.shape[0]):
            l_ub[:] = 0.0
            l_lb[:] = 0.0
            pos = np.nonzero(A[j][idx_unsure] > 0)[0]
            neg = np.nonzero(A[j][idx_unsure] < 0)[0]
            idx_unsure_pos = idx_unsure[pos]
            idx_unsure_neg = idx_unsure[neg]
            l_ub[idx_unsure_pos] = LBs[i][idx_unsure_pos]
            l_lb[idx_unsure_neg] = LBs[i][idx_unsure_neg]
            UB_final[j] -= np.dot(A[j], l_ub)
            LB_final[j] -= np.dot(A[j], l_lb)
        # compute A for next loop
        if i != 1:
            A = np.dot(A, Ws[i-1] * diags[i-1])
        else:
            A = np.dot(A, Ws[i-1]) # diags[0] is 1
    # after the loop is done we get A0
    UB_final += constants
    LB_final += constants

    # step 6: bounding A0 * x
    x_UB = np.empty_like(UBs[0])
    x_LB = np.empty_like(LBs[0])
    
    Ax0 = np.dot(A,x0)
    
    if p_n == 105: # means p == "i":      
        for j in range(A.shape[0]):        
            dualnorm_Aj = np.sum(np.abs(A[j])) # L1 norm of A[j]
            UB_final[j] += (Ax0[j]+eps*dualnorm_Aj)
            LB_final[j] += (Ax0[j]-eps*dualnorm_Aj)

    elif p_n == 1: # means p == "1"
        for j in range(A.shape[0]):        
            dualnorm_Aj = np.max(np.abs(A[j])) # Linf norm of A[j]
            UB_final[j] += (Ax0[j]+eps*dualnorm_Aj)
            LB_final[j] += (Ax0[j]-eps*dualnorm_Aj)

    elif p_n == 2: # means p == "2"
        for j in range(A.shape[0]):        
            dualnorm_Aj = np.linalg.norm(A[j]) # L2 norm of A[j]
            UB_final[j] += (Ax0[j]+eps*dualnorm_Aj)
            LB_final[j] += (Ax0[j]-eps*dualnorm_Aj)
    
    return UB_final, LB_final

# W2 \in [c, M2], W1 \in [M2, M1]
# c, l, u \in [c, M1]
# r \in [c], k \in [M1], i \in [M2]
@jit(nopython=True)
def fast_compute_max_grad_norm_2layer(W2, W1, neuron_state, norm = 1):
    # even if q_n != 1, then algorithm is the same. The difference is only at the output of fast_compute_max_grad_norm
    assert norm == 1
    # diag = 1 when neuron is active
    diag = np.maximum(neuron_state.astype(np.float32), 0)
    unsure_index = np.nonzero(neuron_state == 0)[0]
    # this is the constant part
    c = np.dot(diag * W2, W1)
    # this is the delta, and l <=0, u >= 0
    l = np.zeros((W2.shape[0], W1.shape[1]))
    u = np.zeros_like(l)
    for r in range(W2.shape[0]):
        for k in range(W1.shape[1]):
            for i in unsure_index:
                prod = W2[r,i] * W1[i,k]
                if prod > 0:
                    u[r,k] += prod
                else:
                    l[r,k] += prod
    return c, l, u
    
# prev_c is the constant part; prev_l <=0, prev_u >= 0
# prev_c, prev_l, prev_u \in [c, M2], W1 \in [M2, M1]
# r \in [c], k \in [M1], i \in [M2]
@jit(nopython=True)
def fast_compute_max_grad_norm_2layer_next(prev_c, prev_l, prev_u, W1, neuron_state, norm = 1):
    active_or_unsure_index = np.nonzero(neuron_state >= 0)[0]
    # prev_c is the fix term, direct use 2-layer bound results
    c, l, u = fast_compute_max_grad_norm_2layer(prev_c, W1, neuron_state)
    # now deal with prev_l <= delta <= prev_u term
    # r is dimention for delta.shape[0]
    for r in range(prev_l.shape[0]):
        for k in range(W1.shape[1]):
            for i in active_or_unsure_index:
                if W1[i,k] > 0:
                    u[r,k] += prev_u[r,i] * W1[i,k] 
                    l[r,k] += prev_l[r,i] * W1[i,k]
                else:
                    u[r,k] += prev_l[r,i] * W1[i,k] 
                    l[r,k] += prev_u[r,i] * W1[i,k]
    return c, l, u

#@jit(nopython=True)
def fast_compute_max_grad_norm(weights, neuron_states, numlayer, norm):
    assert numlayer >= 2
    # merge the last layer weights according to c and j
    # W_vec = np.expand_dims(weights[-1][c] - weights[-1][j], axis=0)
    # const, l, u = fast_compute_max_grad_norm_2layer(W_vec, weights[-2], neuron_states[-1])
    const, l, u = fast_compute_max_grad_norm_2layer(weights[-1], weights[-2], neuron_states[-1])
    # for layers other than the last two layers
    for i in list(range(numlayer - 2))[::-1]:
        const, l, u = fast_compute_max_grad_norm_2layer_next(const, l, u, weights[i], neuron_states[i])
    # get the final upper and lower bound
    l += const
    u += const
    l = np.abs(l)
    u = np.abs(u)
    
    max_l_u = np.maximum(l, u)
    #print("max_l_u.shape = {}".format(max_l_u.shape))
    #print("max_l_u = {}".format(max_l_u))
    
    if norm == 1: # q_n = 1, return L1 norm of max component
        return np.sum(max_l_u, axis = 1)
    elif norm == 2: # q_n = 2, return L2 norm of max component
        return np.sqrt(np.sum(max_l_u**2, axis = 1))
    elif norm == 105: # q_n = ord('i'), return Li norm of max component
        # important: return type should be consistent with other returns 
        # For other 2 statements, it will return an array: [val], so we need to return an array.
        # numba doesn't support np.max and list, but support arrays
        
        
        max_ele = np.zeros((max_l_u.shape[0],))
        for i in range(max_l_u.shape[0]):
            for ii in range(max_l_u.shape[1]):
                if max_l_u[i][ii] > max_ele[i]:
                    max_ele[i] = max_l_u[i][ii]
        return max_ele

        # previous code
        #max_ele = np.array([0.0])
        #for i in range(len(max_l_u[0])):
        #    if max_l_u[0][i] > max_ele[0]:
        #        max_ele[0] = max_l_u[0][i]                
        #  
        #return max_ele

@jit(nopython=True)
def inc_counter(layer_counter, weights, layer):
    layer_counter[layer] += 1
    # we don't include the last layer, which does not have activation
    n_layers = len(weights) - 1;
    if layer == n_layers:
        # okay, we have enumerated all layers, and now there is an overflow
        return
    if layer_counter[layer] == weights[n_layers - layer - 1].shape[0]:
        # enumerated all neurons for this layer, increment counter for the next layer
        layer_counter[layer] = 0
        inc_counter(layer_counter, weights, layer+1)

@jit(nopython=True)
def compute_max_grad_norm(weights, c, j, neuron_states, numlayer, norm = 1):
    # layer_counter is the counter for our enumeration progress
    # first element-> second last layer, last elements-> first layershow_histogram
    # extra element to detect overflow (loop ending)
    layer_counter = np.zeros(shape=numlayer, dtype=np.uint16)
    # this is the part 1 of the bound, accumulating all the KNOWN activations
    known_w = np.zeros(weights[0].shape[1])
    # this is the part 2 of the bound, accumulating norms of all unsure activations
    unsure_w_norm = 0.0
    # s keeps the current activation pattern (last layer does not have activation)
    s = np.empty(shape=numlayer - 1, dtype=np.int8)
    # some stats
    skip_count = fixed_paths = unsure_paths = total_loop = 0
    # we will go over ALL possible activation combinations
    while layer_counter[-1] != 1:
        for i in range(numlayer-1):
            # note that layer_counter is organized in the reversed order
            s[i] = neuron_states[i][layer_counter[numlayer - i - 2]]
        # now s contains the states of each neuron we are currently investigating in each layer
        # for example, for a 4-layer network, s could be [-1, 0, 1], means the first layer neuron 
        # no. layer_counter[2] is inactive (-1), second layer neuron no. layer_counter[1] has
        # unsure activation, third layer neuron no. layer_counter[0] is active (1)
        skip = False
        for i in range(numlayer-1): 
            # if any neuron is -1, we skip the entire search range!
            # we look for inactive neuron at the first layer first;
            # we can potentially skip large amount of searches
            if s[i] == -1:
                inc_counter(layer_counter, weights, numlayer - i - 2)
                skip = True
                skip_count += 1
                break
        if not skip:
            total_loop += 1
            # product of all weight parameters
            w = 1.0
            for i in range(0, numlayer-2):
                # product of all weights along the way
                w *= weights[i+1][layer_counter[numlayer - (i+1) - 2], layer_counter[numlayer - i - 2]]
            if np.sum(s) == numlayer - 1:
                fixed_paths += 1
                # all neurons in this path are known to be active.
                known_w += (weights[-1][c,layer_counter[0]] - weights[-1][j,layer_counter[0]]) * w \
                           * weights[0][layer_counter[numlayer - 2]]
            else:
                unsure_paths += 1
                # there must be some neurons have unsure states;
                unsure_w_norm += np.linalg.norm((weights[-1][c,layer_counter[0]] - weights[-1][j,layer_counter[0]]) * w \
                                 * weights[0][layer_counter[numlayer - 2]], norm)
            # increment the counter by 1
            inc_counter(layer_counter, weights, 0)

    known_w_norm = np.linalg.norm(known_w, norm)
    # return the norm and some statistics
    return np.array([known_w_norm + unsure_w_norm]), total_loop, skip_count, fixed_paths, unsure_paths, known_w_norm, unsure_w_norm

"""
Main computing function 2
"""

def compute_worst_bound_multi(weights, biases, pred_label, target_label, x0, predictions, numlayer, p = "i", eps = 0.005, steps = 20, method = "ours", lipsbnd="fast", untargeted = False):
    budget = None
    for e in np.linspace(eps/steps, eps, steps):
        _, g_x0, max_grad_norm = compute_worst_bound(weights, biases, pred_label, target_label, x0, predictions, numlayer, p, e, method, lipsbnd, False, False, untargeted)
        if budget is None:
            budget = g_x0
        new_budget = budget - max_grad_norm * (eps / steps)
        if untargeted:
            for j in range(weights[-1].shape[0]):
                if j < pred_label:
                    print("[L2] j = {}, validating eps={:.4f}, budget={:.4f}, new_budget={:.4f}, max_grad_norm={:.4f}".format(j, e, budget[j], new_budget[j], max_grad_norm[j]))
                elif j > pred_label:
                    print("[L2] j = {}, validating eps={:.4f}, budget={:.4f}, new_budget={:.4f}, max_grad_norm={:.4f}".format(j, e, budget[j-1], new_budget[j-1], max_grad_norm[j-1]))
        else:
            print("[L2] validating eps={:.4f}, budget={:.4f}, new_budget={:.4f}, max_grad_norm={:.4f}".format(e, budget[0], new_budget[0], max_grad_norm[0]))
        budget = new_budget
        if any(budget < 0):
            print("[L2][verification failure] min_perturbation = {:.4f}".format(e - eps/steps))
            return e - eps/steps
    print("[L2][verification success] eps = {:.4f}".format(e))
    return eps



"""
Main computing function 1
"""
def compute_worst_bound(weights, biases, pred_label, target_label, x0, predictions, numlayer, p="i", eps = 0.005, method="ours", lipsbnd="disable", is_LP=False, is_LPFULL=False,untargeted = False):
    ### input example x0 
    # 784 by 1 (cifar: 3072 by 1)
    x0 = x0.flatten().astype(np.float32)
    # currently only supports p = "i"
    UB_N0 = x0 + eps
    LB_N0 = x0 - eps
    
    # convert p into numba compatible form
    if p == "i":
        p_n = ord('i') # 105
        q_n = 1 # the grad_norm 
    elif p == "1":
        p_n = 1
        q_n = ord('i') # 105
    elif p == "2":
        p_n = 2
        q_n = 2
    else:
        print("currently doesn't support p = {}, only support p = i,1,2".format(p))
    
    # contains numlayer+1 arrays, each corresponding to a lower/upper bound 
    UBs = []
    LBs = []
    UBs.append(UB_N0)
    LBs.append(LB_N0)
    #save_bnd = {'UB_N0': UB_N0, 'LB_N0': LB_N0}
    neuron_states = []
    
       
    c = pred_label # c = 0~9
    j = target_label 

    # create diag matrices
    diags = init_layer_bound_relax_matrix_huan(weights)
    
    ## weights and biases are already transposed
    # output of get_layer_bound can be raw UB/LB or ReLU(UB/LB) by assigning the flag
    # output of get_layer_bound_relax is raw UB/LB
    if method == "naive": # simplest worst case bound
        for num in range(numlayer):
            W = weights[num] 
            bias = biases[num]
            
            # middle layer
            if num < numlayer-1:
                UB, LB = get_layer_bound(W,bias,UBs[num],LBs[num],False)
                neuron_states.append(np.zeros(shape=bias.shape, dtype=np.int8))
                # neurons never activated set to -1
                neuron_states[-1] -= UB == 0
                # neurons always activated set to +1
                neuron_states[-1] += LB > 0
                # other neurons could be activated or inactivated
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,", 
                                    sum(neuron_states[-1] == +1), "neurons always activated")
            else: # last layer
                UB, LB = get_layer_bound(W,bias,UBs[num],LBs[num],True)        
                        
            UBs.append(UB)
            LBs.append(LB)
            
    elif method == "ours" or is_LPFULL:
        # contains numlayer arrays, each corresponding to a pre-ReLU bound
        preReLU_UB = []
        preReLU_LB = []
        
        for num in range(numlayer):
            
            # first time compute the bound of 1st layer
            if num == 0: # get the raw bound
                if is_LPFULL:
                    UB, LB, _ = get_layer_bound_LP(weights[:num+1],biases[:num+1],[UBs[0]],
                    [LBs[0]],x0,eps,p,neuron_states,1,c,j,True,False)
                    #UB, LB = get_layer_bound(weights[num],biases[num],UBs[num],LBs[num],True)
                else:
                    UB, LB = get_layer_bound(weights[num],biases[num],UBs[num],LBs[num],True,x0,eps,p_n)
                # save those pre-ReLU bounds
                preReLU_UB.append(UB)
                preReLU_LB.append(LB)
                # apply ReLU here manually
                UB = ReLU(UB)
                LB = ReLU(LB)

                neuron_states.append(np.zeros(shape=biases[num].shape, dtype=np.int8))
                # neurons never activated set to -1
                neuron_states[-1] -= UB == 0
                # neurons always activated set to +1
                neuron_states[-1] += LB > 0
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,", 
                                    sum(neuron_states[-1] == +1), "neurons always activated")                               
                UBs.append(UB)
                LBs.append(LB)              
                
            # we skip the last layer, which will be dealt later
            elif num != numlayer - 1:
                
                if is_LPFULL:
                    UB, LB, _ = get_layer_bound_LP(weights[:num+1],biases[:num+1],[UBs[0]]+preReLU_UB,
                    [LBs[0]]+preReLU_LB,x0,eps,p,neuron_states,num+1,c,j,True,False)
                else:
                    # UB, LB = get_layer_bound_relax_matrix_huan(weights[:num+1],biases[:num+1],

                    UB, LB = get_layer_bound_relax_matrix_huan_optimized(tuple(weights[:num+1]),tuple(biases[:num+1]),
                    tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB), 
                    tuple(neuron_states),
                    num + 1,tuple(diags[:num+1]),
                    x0,eps,p_n)
                # last layer has no activation
                # save those pre-ReLU bounds
                preReLU_UB.append(UB)
                preReLU_LB.append(LB)
                # apply ReLU here manually
                UB = ReLU(UB)
                LB = ReLU(LB)
                # Now UB and LB act just like before
                neuron_states.append(np.zeros(shape=biases[num].shape, dtype=np.int8))
                # neurons never activated set to -1
                neuron_states[-1] -= UB == 0
                # neurons always activated set to +1
                neuron_states[-1] += LB > 0
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,", 
                                    sum(neuron_states[-1] == +1), "neurons always activated")
                UBs.append(UB)
                LBs.append(LB)

    else:
        raise(RuntimeError("unknown method number: {}".format(method)))
                
        
    num = numlayer - 1
    W = weights[num]
    bias = biases[num]
    if untargeted:
        ind = np.ones(len(W), bool)
        ind[c] = False
        W_last = W[c] - W[ind]
        b_last = bias[c] - bias[ind]
    else:
        W_last = np.expand_dims(W[c] - W[j], axis=0)
        b_last = np.expand_dims(bias[c] - bias[j], axis=0)
    if method == "naive":
        UB, LB = get_layer_bound(W_last,b_last,UB,LB,True)
    elif method == "ours":
        # UB, LB = get_layer_bound_relax_matrix_huan(weights[:num]+[W_last],biases[:num]+[b_last],
        #            [UBs[0]]+preReLU_UB,[LBs[0]]+preReLU_LB,
        #            neuron_states,
        #            True, numlayer)

        UB, LB = get_layer_bound_relax_matrix_huan_optimized(tuple(weights[:num]+[W_last]),tuple(biases[:num]+[b_last]),
        tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB),
        tuple(neuron_states),
        numlayer,tuple(diags),
        x0,eps,p_n)


    # Print bounds results
    print("epsilon = {:.5f}".format(eps))
    # print("c = {}, {:.2f} < f_c < {:.2f}".format(c, LBs[numlayer][c], UBs[numlayer][c]))
    # print("j = {}, {:.2f} < f_j < {:.2f}".format(j, LBs[numlayer][j], UBs[numlayer][j])) 
    
    # after all bounds has been computed, we run a LP to find the last layer bounds
    if is_LP or is_LPFULL:
        if untargeted:
            LP_UB, LP_LB, LP_LBs = get_layer_bound_LP(weights,biases,[UBs[0]]+preReLU_UB,
            [LBs[0]]+preReLU_LB, x0, eps, p, neuron_states,numlayer,c,j,False,True)
            LP_UBs = np.empty_like(LP_LBs)
            LP_UBs[:] = np.inf
        else:
            LP_UB, LP_LB, LP_bnd_gx0 = get_layer_bound_LP(weights,biases,[UBs[0]]+preReLU_UB,
            [LBs[0]]+preReLU_LB, x0, eps, p, neuron_states,numlayer,c,j,False,False)
        # print("c = {}, {:.2f} < f_c < {:.2f}".format(c, LP_LB[c], LP_UB[c]))
        # print("j = {}, {:.2f} < f_j < {:.2f}".format(j, LP_LB[j], LP_UB[j])) 
        
    if untargeted:
        for j in range(W.shape[0]):
            if j < c:
                print("    {:.2f} < f_c - f_{} < {:.2f}".format(LB[j], j, UB[j]))
                if is_LP:
                    print("LP  {:.2f} < f_c - f_{} < {:.2f}".format(LP_LBs[j], j, LP_UBs[j]))
            elif j > c:
                print("    {:.2f} < f_c - f_{} < {:.2f}".format(LB[j-1], j, UB[j-1]))
                if is_LP:
                    print("LP  {:.2f} < f_c - f_{} < {:.2f}".format(LP_LBs[j-1], j, LP_UBs[j-1]))
        if is_LP or is_LPFULL:
            gap_gx = np.min(LP_LBs)
        else:
            gap_gx = np.min(LB)
    else:
        if is_LP or is_LPFULL:
            print("    {:.2f} < f_c - f_j ".format(LP_bnd_gx0))
            gap_gx = LP_bnd_gx0
        else:
            print("    {:.2f} < f_c - f_j < {:.2f}".format(LB[0], UB[0]))
            gap_gx = LB[0]

    # Now "weights" are already transposed, so can pass weights directly to compute_max_grad_norm. 
    # Note however, if we transpose weights again, compute_max_grad_norm still works, but the result is different   
    # compute lipschitz bound
    if untargeted:
        g_x0 = []
        for j in range(W.shape[0]):
            if j < c:
                g_x0.append(predictions[c] - predictions[j])
            elif j > c:
                g_x0.append(predictions[c] - predictions[j])
    else:
        g_x0 = [predictions[c] - predictions[j]]
    max_grad_norm = 0.0
    if lipsbnd == "both" or lipsbnd == "naive": 
        assert untargeted == False
        norm = 2
        if p == "i":
            norm = 1
        if p == "1":
            norm = np.inf
        max_grad_norm, total_loop, skip_count, fixed_paths, unsure_paths, known_w_norm, unsure_w_norm = compute_max_grad_norm(
                    tuple(weights), c, j, tuple(neuron_states), numlayer, norm)
        print("total_loop={}, skipped_loop={}, known_activations={}, unsure_activations={}, known_norm={}, unsure_norm={}".format(total_loop, skip_count, fixed_paths, unsure_paths, known_w_norm, unsure_w_norm))
        print("g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(g_x0[0], max_grad_norm[0], g_x0[0] / max_grad_norm[0]))
    if lipsbnd == "both" or lipsbnd == "fast":
        max_grad_norm = fast_compute_max_grad_norm(tuple(weights[:num]+[W_last]), tuple(neuron_states), numlayer, q_n)
        if untargeted:
            for j in range(W.shape[0]):
                if j < c:
                    print("j = {}, g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(j, g_x0[j], max_grad_norm[j], g_x0[j] / max_grad_norm[j]))
                elif j > c:
                    #print("j = {}".format(j))
                    #print("g_x0.shape = {}, max_grad_norm.shape = {}".format(len(g_x0),len(max_grad_norm)))
                    print("j = {}, g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(j, g_x0[j-1], max_grad_norm[j-1], g_x0[j-1] / max_grad_norm[j-1]))
        else:
            print("g_x0 = {:.4f}, lipschitz = {:.4f}, bnd = {:.5f}".format(g_x0[0], max_grad_norm[0], g_x0[0] / max_grad_norm[0]))

    return gap_gx, g_x0, max_grad_norm


