import numpy as np

def convert_fp_to_embV2(vector, size):
    """
    Converts a fingerprint vector into a embedded vector using a specified size.
    Fingerprint shape must be N, M*size, Where N is the number of Fingerprints of
    N molecules and M*size means that the length of the fingerprint must be an 
    integer multiple of size

    Args:
        vector (numpy.ndarray): Input fingerprint vector.
        size (int): Size used for reshaping the fingerprint vector.

    Returns:
        numpy.ndarray: Embedded vector obtained by reshaping and processing the input fingerprint vector.

    Raises:
        ValueError: If the size is not a positive integer.
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer. Try size = 16, or size = 32")
    
    if len(vector.shape) == 1:
        rows = 1 
        cols = vector.shape[0]
    else:
        rows, cols = vector.shape

    narrays = cols // size

    mask = 2**np.arange(size, dtype = np.float64 )
    mask = mask/np.sum(mask)
    bigMask = np.tile(mask, (narrays, 1))

    tensorMask = np.tile(bigMask, (rows, 1))
    tensorMask = tensorMask.reshape((rows* narrays, size))
    vector_reshape = vector.reshape((rows* narrays, size))
    
    mfp_masked = tensorMask * vector_reshape
    mfp_maskedDotted = np.sum(mfp_masked, axis=1)
    
    if len(vector.shape) == 1:
        return mfp_maskedDotted.reshape((rows,narrays))
    else:
        return mfp_maskedDotted.reshape((rows,narrays)).squeeze()
