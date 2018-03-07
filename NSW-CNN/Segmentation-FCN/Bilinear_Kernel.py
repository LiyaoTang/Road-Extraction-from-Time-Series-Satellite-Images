import numpy as np

# kernel size for bilinear interpolation
def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def get_bilinear_upsample_weights(factor, in_channel, out_channel):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size, # height
                        filter_size, # width
                        out_channel,
                        in_channel), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for in_idx in range(in_channel):
        for out_idx in range(out_channel):
            weights[:, :, out_idx, in_idx] = upsample_kernel
    
    return weights