from cnn_gp import Sequential, Conv2d, ReLU, NormalizationModule
import torch
import numpy as np


def calc_kernel(model, data,num_in_split=200):
    """Splits self kernel calulation in multiple patches to reduce computational load.

    Args:
        model (cnn_gp.Sequential): Gaussian process
        data (torch.tensor): training data
        num_in_split (int): number of splitting of training data. Defaults to 200.

    Returns:
        (torch.tensor): full kernel matrix
    """
    n = int((len(data)-1)/num_in_split)+1
    num_right_mats = (n-1)*n/2
    data = data.double()
    data_list = [data[i*num_in_split:(i+1)*num_in_split] for i in range(n)]
    result = model(data_list[0])
    for i in range(1,n):
        right = [model(data_list[i-pos],data_list[i]) for pos in range(i,0,-1)]
        middle = model(data_list[i],data_list[i])
        bottom = [torch.t(x) for x in right]
        right.append(middle)
        result = torch.cat([torch.cat([result,torch.cat(bottom, dim=1)],dim=0),torch.cat(right,dim=0)],dim=1)

    return result.double()

def calc_cross_kernel(model, X,Y,num_in_split=200):
    """Splits cross kernel calulation in multiple patches to reduce computational load.

    Args:
        model (cnn_gp.Sequential): Gaussian process
        X (torch.tensor): training data
        X (torch.tensor): test data
        num_in_split (int): number of splitting of training data. Defaults to 200.

    Returns:
        (torch.tensor): full kernel matrix
    """    
    n_x = int((len(X)-1)/num_in_split)+1
    n_y = int((len(Y)-1)/num_in_split)+1
    X = X.double()
    Y = Y.double()
    X_list  = [X[i*num_in_split:(i+1)*num_in_split] for i in range(n_x)]
    Y_list  = [Y[i*num_in_split:(i+1)*num_in_split] for i in range(n_y)]
    
    Kvs = []
    for x in X_list:
        Khs = [model(x,y) for y in Y_list]
        Kvs.append(torch.cat(Khs,dim=1))
    result = torch.cat(Kvs,dim = 0)
    return result.double()


def generate_model(num_layers,relu = True, normalize=False,var_bias = 0., var_weight = 1.,kernel_size=7):
    """Generates default model

    Args:
        num_layers (_type_): _description_
        relu (bool, optional): _description_. Defaults to True.
        normalize (bool, optional): _description_. Defaults to False.
        var_bias (_type_, optional): _description_. Defaults to 0..
        var_weight (_type_, optional): _description_. Defaults to 1..
        kernel_size (int, optional): _description_. Defaults to 7.

    Returns:
        _type_: _description_
    """
    # var_bias = 7.86
    # var_weight = 2.79
    modules = []
    for i in range(num_layers):
        modules.append(Conv2d(kernel_size=kernel_size, padding="same", var_weight=var_weight*kernel_size**2, var_bias=var_bias))
        if relu:
            modules.append(ReLU())
        if normalize:
            modules.append(NormalizationModule())
    return Sequential(*modules, Conv2d(kernel_size=28, padding=0, var_weight=var_weight*28**2, var_bias=var_bias),).double()