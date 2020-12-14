# -*- coding: utf-8 -*-
# ncon.py
import jax.numpy as np
import numpy as onp

def jncon(tensor_list, connect_list_in, cont_order=None, check_network=True):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.31) - last modified 30/8/2019
------------------------
Network CONtractor. Input is an array of tensors 'tensor_list' and an array \
of vectors 'connect_list_in', with each vector labelling the indices of the \
corresponding tensor. Labels should be  positive integers for contracted \
indices and negative integers for free indices. Optional input 'cont_order' \
can be used to specify order of index contractions (otherwise defaults to \
ascending order of the positive indices). Checking of the consistancy of the \
input network can be disabled for slightly faster operation.

Further information can be found at: https://arxiv.org/abs/1402.0939
    """

    # put inputs into a list if necessary
    if type(tensor_list) is not list:
        tensor_list = [tensor_list]
    if type(connect_list_in[0]) is not list:
        connect_list_in = [connect_list_in]
    connect_list = [0 for x in range(len(connect_list_in))]
    for ele in range(len(connect_list_in)):
        connect_list[ele] = onp.array(connect_list_in[ele])

    # generate contraction order if necessary
    flat_connect = onp.array([item for sublist in connect_list for item in sublist])
    if cont_order == None:
        cont_order = onp.unique(flat_connect[flat_connect > 0])
    else:
        cont_order = onp.array(cont_order)

    # check inputs if enabled
    if check_network:
        dims_list = [list(tensor.shape) for tensor in tensor_list]
        check_inputs(connect_list, flat_connect, dims_list, cont_order)

    # do all partial traces
    for ele in range(len(tensor_list)):
        num_cont = len(connect_list[ele]) - len(onp.unique(connect_list[ele]))
        if num_cont > 0:
            tensor_list[ele], connect_list[ele], cont_ind = partial_trace(tensor_list[ele], connect_list[ele])
            cont_order = onp.delete(cont_order, onp.intersect1d(cont_order,cont_ind,return_indices=True)[1])

    # do all binary contractions
    while len(cont_order) > 0:
        # identify tensors to be contracted
        cont_ind = cont_order[0]
        locs = [ele for ele in range(len(connect_list)) if sum(connect_list[ele] == cont_ind) > 0]

        # do binary contraction
        cont_many, A_cont, B_cont = onp.intersect1d(connect_list[locs[0]], connect_list[locs[1]], assume_unique=True, return_indices=True)
        tensor_list.append(np.tensordot(tensor_list[locs[0]], tensor_list[locs[1]], axes=( list(A_cont), list(B_cont) ) ) )
        connect_list.append(onp.append(onp.delete(connect_list[locs[0]], A_cont), onp.delete(connect_list[locs[1]], B_cont)))

        # remove contracted tensors from list and update cont_order
        del tensor_list[locs[1]]
        del tensor_list[locs[0]]
        del connect_list[locs[1]]
        del connect_list[locs[0]]
        cont_order = onp.delete(cont_order,onp.intersect1d(cont_order,cont_many, assume_unique=True, return_indices=True)[1])

    # do all outer products
    while len(tensor_list) > 1:
        s1 = tensor_list[-2].shape
        s2 = tensor_list[-1].shape
        tensor_list[-2] = np.outer(tensor_list[-2].reshape(onp.prod(s1)),
                   tensor_list[-1].reshape(onp.prod(s2))).reshape(onp.append(s1,s2))
        connect_list[-2] = onp.append(connect_list[-2],connect_list[-1])
        del tensor_list[-1]
        del connect_list[-1]

    # do final permutation
    if len(connect_list[0]) > 0:
        return np.transpose(tensor_list[0],onp.argsort(-connect_list[0]))
    else:
        return tensor_list[0]

#-----------------------------------------------------------------------------
def partial_trace(A, A_label):
    """ Partial trace on tensor A over repeated labels in A_label """

    num_cont = len(A_label) - len(onp.unique(A_label))
    if num_cont > 0:
        dup_list = []
        for ele in onp.unique(A_label):
            if sum(A_label == ele) > 1:
                dup_list.append([onp.where(A_label == ele)[0]])

        cont_ind = onp.array(dup_list).reshape(2*num_cont,order='F')
        free_ind = onp.delete(onp.arange(len(A_label)),cont_ind)

        cont_dim = onp.prod(onp.array(A.shape)[cont_ind[:num_cont]])
        free_dim = onp.array(A.shape)[free_ind]

        B_label = onp.delete(A_label, cont_ind)
        cont_label = onp.unique(A_label[cont_ind])
        B = onp.zeros(onp.prod(free_dim))
        A = A.transpose(onp.append(free_ind, cont_ind)).reshape(onp.prod(free_dim),cont_dim,cont_dim)
        for ip in range(cont_dim):
            B = B + A[:,ip,ip]

        return B.reshape(free_dim), B_label, cont_label

    else:
        return A, A_label, []

#-----------------------------------------------------------------------------
def check_inputs(connect_list, flat_connect, dims_list, cont_order):
    """ Check consistancy of NCON inputs"""

    pos_ind = flat_connect[flat_connect > 0]
    neg_ind = flat_connect[flat_connect < 0]

    # check that lengths of lists match
    if len(dims_list) != len(connect_list):
        raise ValueError(('NCON error: %i tensors given but %i index sublists given')
            %(len(dims_list), len(connect_list)))

    # check that tensors have the right number of indices
    for ele in range(len(dims_list)):
        if len(dims_list[ele]) != len(connect_list[ele]):
            raise ValueError(('NCON error: number of indices does not match number of labels on tensor %i: '
                              '%i-indices versus %i-labels')%(ele,len(dims_list[ele]),len(connect_list[ele])))

    # check that contraction order is valid
    if not onp.array_equal(onp.sort(cont_order),onp.unique(pos_ind)):
        raise ValueError(('NCON error: invalid contraction order'))

    # check that negative indices are valid
    for ind in onp.arange(-1,-len(neg_ind)-1,-1):
        if sum(neg_ind == ind) == 0:
            raise ValueError(('NCON error: no index labelled %i') %(ind))
        elif sum(neg_ind == ind) > 1:
            raise ValueError(('NCON error: more than one index labelled %i')%(ind))

    # check that positive indices are valid and contracted tensor dimensions match
    flat_dims = onp.array([item for sublist in dims_list for item in sublist])
    for ind in onp.unique(pos_ind):
        if sum(pos_ind == ind) == 1:
            raise ValueError(('NCON error: only one index labelled %i')%(ind))
        elif sum(pos_ind == ind) > 2:
            raise ValueError(('NCON error: more than two indices labelled %i')%(ind))

        cont_dims = flat_dims[flat_connect == ind]
        if cont_dims[0] != cont_dims[1]:
            raise ValueError(('NCON error: tensor dimension mismatch on index labelled %i: '
                              'dim-%i versus dim-%i')%(ind,cont_dims[0],cont_dims[1]))

    return True
#-----------------------------------------------------------------------------
