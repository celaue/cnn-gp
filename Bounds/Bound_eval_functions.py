import numpy as np
import math
from utils_bounds import calc_kernel_selector


def calc_bound(l,X,Y,sig_w=1,W=calc_kernel_selector()):
    """Calculates both terms of bounds for upper and lower bound for ConvNet-GP-L following equations (36,37) without prefactors.  
    Args:
        l (int): number of layers
        X (np.array): X
        Y (np.array): X^\prime
        sig_w (float): variance. Defaults to 1.
        W (np.array): Toeplitz matrix. Defaults to calc_kernel_selector().

    Returns:
        result (np.array): both terms (T1, T2) without prefactor
    """
    W_l = np.ones(784)*sig_w
    Ws = np.zeros((l,784,784))
    Ws[0]=W*sig_w
    for i in range(1,l):
        Ws[i] = W@Ws[i-1]*sig_w 
    Nx = X.size(0)
    Ny = Y.size(0)
    result = np.ones((2,Nx,Ny))
    
    for ix in range(Nx):
        for iy in range(Ny):
            x = X[ix].view(-1,1).double().numpy()
            y = Y[iy].view(-1,1).double().numpy()

            Pxy = np.sqrt(Ws[l-1]@(x*x))*np.sqrt(Ws[l-1]@(y*y))
            T1 = W_l@Pxy/2
            for k in range(2,l+1):
                Pxy = np.sqrt(Ws[l-k]@(x*x))*np.sqrt(Ws[l-k]@(y*y))  
                T1 = T1 + (W_l@Ws[k-2])@Pxy/2**k
            T1 = T1*(1./2.)**(l)
            T2 = (1./4.)**(l)*W_l@Ws[l-1]@(x*y)
            result[0,ix,iy] = T1
            result[1,ix,iy] = T2
            
    return result

def calc_upper_bound(l,X,Y,sig_w=1,W=calc_kernel_selector()):
    """Calculates upper bound  
    Args:
        l (int): number of layers
        X (np.array): X
        Y (np.array): X^\prime
        sig_w (float): variance. Defaults to 1.
        W (np.array): Toeplitz matrix. Defaults to calc_kernel_selector().

    Returns:
        result (float): upper bound
    """
    zw = calc_bound(l,X,Y,sig_w,W)
    return zw[0]+zw[1]

def calc_lower_bound(l,X,Y,sig_w=1,W=calc_kernel_selector()):
    """Calculates lower bound  
    Args:
        l (int): number of layers
        X (np.array): X
        Y (np.array): X^\prime
        sig_w (float): variance. Defaults to 1.
        W (np.array): Toeplitz matrix. Defaults to calc_kernel_selector().

    Returns:
        result (float): lower bound
    """
    zw = calc_bound(l,X,Y,sig_w,W)
    return zw[0]*2/math.pi+zw[1]
