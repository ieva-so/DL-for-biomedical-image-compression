import numpy as np

def get_psnr(original_img_arr, decoded_img_arr, ignore=None):
    # Calculate the maximum data value
    maximumDataValue = np.maximum(np.amax(original_img_arr), np.amax(decoded_img_arr))
    d1 = original_img_arr.flatten()
    d2 = decoded_img_arr.flatten()

    # Make sure that the provided data sets are the same size
    if d1.size != d2.size:
        raise ValueError('Provided datasets must have the same size/shape')

    # Check if the provided data sets are identical, and if so, return an
    # infinite peak-signal-to-noise ratio
    if np.array_equal(d1, d2):
        return float('inf')

    # If specified, remove the values to ignore from the analysis and compute
    # the element-wise difference between the data sets
    if ignore is not None:
        index = np.intersect1d(np.where(d1 != ignore)[0], 
                                    np.where(d2 != ignore)[0])
        error = d1[index].astype(np.float64) - d2[index].astype(np.float64)
    else:
        error = d1.astype(np.float64)-d2.astype(np.float64)

    # Compute the mean-squared error
    meanSquaredError = np.sum(error**2) / error.size

    # Return the peak-signal-to-noise ratio
    return 10.0 * np.log10(maximumDataValue**2 / meanSquaredError)