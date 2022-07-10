import numpy as np

def determine_no_pad_pos(in_size,true_in_size):
    """Determines all position in teoplitz matrix not corresponding to zeropadding components

    Args:
        in_size (int,int): size with padding
        true_in_size (int,int): size without padding

    Returns:
        list(int): no_pad postions
    """
    diff = np.array(in_size)-np.array(true_in_size)
    left_top_padding = (diff/2).astype(int)
    result = []
    no_pad = np.zeros(in_size)
    no_pad[left_top_padding[0]:-left_top_padding[0],left_top_padding[1]:-left_top_padding[1]] = np.ones(true_in_size)
    no_pad = no_pad.reshape(-1)
    return list(np.where(no_pad)[0])

def calc_toeplitz(kernel, in_size, out_size,true_in_size=None):
    """Calculates selection toeplitz matrix.

    Args:
        kernel (int,int): kernel size
        in_size (int,int): input size
        out_size (int,int): output size
        true_in_size (int,int): output size without padding. Defaults to None.

    Returns:
       result(np.array): Selction Toeplitz matrix
    """
    out_v = out_size[0]*out_size[1]
    in_v = in_size[0]*in_size[1]
    K = kernel.shape[0]
    result = -np.ones([out_v,in_v])
    in_mat = np.arange(in_v).reshape(in_size)
    kernel_vector = kernel.reshape(1,-1)
    counter = 0
    for r in range(in_size[0]-K+1):
        for c in range(in_size[1]-K+1):
            rel_idx = in_mat[r:r+K,c:c+K].reshape(1,-1)
            result[counter,rel_idx]= kernel_vector
            counter += 1
            
    # ASS: Symmetric padding
    if true_in_size is not None:
        idxs = determine_no_pad_pos(in_size,true_in_size)
        result = result[:,idxs]

    return result

def calc_kernel_selector(kernel=np.ones((7,7)),in_size=(34,34),out_size=(28,28),true_in_size=(28,28)):
    """Calculate constant toeplitz matrix for convolutional layer $W^~$.

    Args:
        kernel (int,int): kernel size
        in_size (int,int): input size
        out_size (int,int): output size
        true_in_size (int,int): output size without padding. 

    Returns:
        result(np.array): Constant Toeplitz matrix
    """
    toep = calc_toeplitz(kernel,in_size,out_size,true_in_size)
    toep[toep==-1] = 0
    toep[toep!=0] = 1
    return toep

