from scipy import linalg
import numpy as np
import torch

def solve_system(Kxx, Y):
    """Determine inverse of self kernel matrix.

    Args:
        Kxx : self kernel matrix
        Y : labels

    Returns:
        (np.array): K^-1.Y
    """
    print("Running scipy solve Kxx^-1 Y routine")
    if type(Kxx)==np.ndarray and type(Y)==np.ndarray:
        
        A = linalg.lstsq(Kxx, Y, cond = 1e-8)[0]
        A = A.astype('float64')
        return A
    elif type(Kxx)!=np.ndarray and type(Y)==np.ndarray or type(Kxx)==np.ndarray and type(Y)!=np.ndarray:
        print("Kxx and Y have different datatype")
        return None
    else:
        assert Kxx.dtype == torch.float64 and Y.dtype == torch.float64
        A = linalg.lstsq(
            Kxx.numpy(), Y.numpy(), cond = 1e-8)[0]
        A = A.astype('float64')
        return torch.from_numpy(A)

def calc_predictions(model,Kxx,Y_train,X_train,Z):
    X_train = X_train.double()
    Z = Z.double()
    A = solve_system(Kxx,Y_train)
    Kzx = calc_cross_kernel(model,Z,X_train)
    return A,(Kzx.double() @ A.double()).argmax(dim=1)


def calc_predictions_only(model,A,X_train,Z):
    X_train = X_train.double()
    Z = Z.double()
    Kzx = calc_cross_kernel(model,Z,X_train)
    return (Kzx.double() @ A.double()).argmax(dim=1)


def calc_accuracy(Y_pred,Y_true):
    return 1.-float(torch.count_nonzero(Y_pred-Y_true)/len(Y_pred))