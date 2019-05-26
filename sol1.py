from skimage import color
from skimage import io
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

GREYSCALE_CODE = 1
RGB_CODE = 2
MAXIMUM_PIXEL_INTENSE = 255
TRANSFORMATION_MATRIX = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.275, -0.321],
    [0.212, -0.523, 0.311]
])
THREE_DIM = 3
TWO_DIM = 2
ERROR_MSG = "Error: the given image has wrong dimensions"

def norm_it(element):
    """
    this function normalizes elements between range 0 and 255 to range 0 to 1
    if the element is negative it makes it zero
    """
    if (element < 0):
        return 0
    else:
        return element / MAXIMUM_PIXEL_INTENSE


def normalize_helper(array):
    """
    this is a helper function to normalize operations
    """
    normal = np.vectorize(norm_it, otypes=[np.float64])
    return normal(array)


def normalize(array):
    """
    this function normalize array values from range between 0 and 255
    to range 0 to 1
    """
    array = array.astype(np.float64)
    array = normalize_helper(array)
    return array.astype(np.float64)


def read_image(filename, representation):
    """
    this function reads an image file and returns it in a given representation
    filename is the image
    representation code: 1 is grayscale, 2 is RGB
    returns an image
    """
    final_img = io.imread(filename)
    if (representation == GREYSCALE_CODE):
        final_img = color.rgb2gray(final_img)
    # if the return wanted image is greyscale so the rgb2gray function does the normalization
    else:
        final_img = normalize(final_img)
    return final_img.astype(np.float64)


def imdisplay(filename, representation):
    """
    this function utilizes the read_image function to display the given image
    in the given representation
    """
    view_img = read_image(filename, representation)
    if (representation == GREYSCALE_CODE):
        plt.imshow(view_img, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(view_img)
    plt.show()


def rgb2yiq(imRGB):
    """
    this function converts RGB image to YIQ
    """
    return np.dot(imRGB, TRANSFORMATION_MATRIX.T)


def yiq2rgb(imYIQ):
    """
    this function converts YIQ image to RGB
    """
    inverse_transform = inv(TRANSFORMATION_MATRIX)
    return np.dot(imYIQ, inverse_transform.T)


def histogram_equalize(im_orig):
    """
    this function performs histogram equalization on a given image input - im_orig
    (greyscale or rgb - in float64 between 0 and 1)
    returns:
        im_eq - the equalized image
        hist_orig - histogram of the original image
        hist_eq - histogram of the equalized image
    """
    is_rgb = False
    cur_img = im_orig.copy()  # Greyscale

    if (im_orig.ndim == THREE_DIM):  # RGB
        im = rgb2yiq(im_orig.copy())
        cur_img = im[:, :, 0]  # yiq - y channel
        is_rgb = True

    elif (im_orig.ndim != TWO_DIM):
        print(ERROR_MSG)
        exit()

    full_range_img = cur_img * MAXIMUM_PIXEL_INTENSE  # from range [0-1] to [0-255]
    hist_orig = np.histogram(full_range_img, np.arange(MAXIMUM_PIXEL_INTENSE + 2))[0]
    cumulative_hist = np.cumsum(hist_orig)
    normed_hist = cumulative_hist / hist_orig.size

    if (np.min(normed_hist) == 0 and np.max(normed_hist) == MAXIMUM_PIXEL_INTENSE):
        rounded_hist = np.around(normed_hist).astype('uint8')
    else:
        cdf_masked = np.ma.masked_equal(normed_hist, 0) # ignore zeroes
        cdf_masked = (cdf_masked - cdf_masked.min()) \
                     / (cdf_masked.max() - cdf_masked.min()) * MAXIMUM_PIXEL_INTENSE
        cdf_masked = np.ma.filled(cdf_masked, 0)
        rounded_hist = np.around(cdf_masked).astype('uint8')

    im_eq = rounded_hist[np.around(full_range_img).astype('uint8')]
    hist_eq = np.histogram(im_eq, np.arange(MAXIMUM_PIXEL_INTENSE + 2))[0]

    if (is_rgb):
        im[:, :, 0] = (im_eq / MAXIMUM_PIXEL_INTENSE).clip(0, 1)
        im_eq = yiq2rgb(im).clip(0, 1).astype(np.float64)
    else:
        im_eq = normalize(im_eq)

    return [im_eq, hist_orig, hist_eq]


def calc_pixel_error(q_i, z_i, z_i_1, hist):
    """
    calc error by pixel
    """
    res_err = 0
    for g in range(z_i, z_i_1 + 1):
        res_err += np.power(q_i - g, 2) * hist[g]
    return res_err


def calc_z(n_quant, q):
    """
    calc z
    """
    z_new = np.zeros(n_quant + 1)
    for t in range(1, len(q)):
        z_new[t] = (q[t - 1] + q[t]) / 2
    z_new[0] = 0
    z_new[-1] = MAXIMUM_PIXEL_INTENSE
    return np.round(z_new).astype("uint8")


def calc_q(z_i, z_i_1, hist):
    """
    calc q
    """
    g_vector = np.arange(z_i, z_i_1 + 1)
    hist_vector = hist[z_i:(z_i_1 + 1)]
    top = np.dot(g_vector, hist_vector)
    bottom = np.sum(hist_vector)
    return np.round(top / bottom)


def quantize(im_orig, n_quant, n_iter):
    """
    this function performs optimal quantization of a given greyscale or RGB image
        im_orig - the input image (float64 values between [0, 1])
        n_quant - the number of intensities for the output
        n_iter - the maximum number of iterations for the procedure
    returns:
        im_quant - quantized image
        error - an array with shape n_iter or less, of the total intensities error for each iteration
    """
    is_rgb = False
    cur_im = im_orig.copy()  # if greyscale

    if (im_orig.ndim == THREE_DIM):  # if RGB
        cur_im = rgb2yiq(im_orig.copy())[:, :, 0]  # y channel of yiq
        is_rgb = True
    elif (im_orig.ndim != TWO_DIM):
        print(ERROR_MSG)
        exit()
    cur_im = np.around(MAXIMUM_PIXEL_INTENSE * cur_im)

    # init borders z (approx. pixel equal division to segments)
    total_pixels = cur_im.size
    num_of_colors = n_quant
    pixels_per_segment = total_pixels / num_of_colors
    hist = np.histogram(cur_im, np.arange(MAXIMUM_PIXEL_INTENSE + 2))[0]
    cdf = np.cumsum(hist)
    z = np.zeros(n_quant + 1)

    for i in range(1, n_quant):
        tmp_idx_arr = np.where(cdf >= i * pixels_per_segment)[0]
        if tmp_idx_arr.size > 0:
            tmp_idx = tmp_idx_arr[0]
            z[i] = tmp_idx
        else:
            z[i] = z[i - 1] if i > 0 else 0
    z[0] = 0
    z[-1] = MAXIMUM_PIXEL_INTENSE
    z = z.astype("uint8")


    cur_q = np.zeros(n_quant) # init the values (q)
    cur_z = z.copy()
    cumulative_pixel_error = 0
    error = []
    last_z = np.zeros(n_quant + 1)
    for k in range(n_iter):
        for j in range(len(cur_q)):
            cur_q[j] = calc_q(cur_z[j], cur_z[j + 1], hist)
            cumulative_pixel_error += calc_pixel_error(cur_q[j], cur_z[j], cur_z[j + 1], hist)
        error.append(cumulative_pixel_error)
        cumulative_pixel_error = 0
        cur_z = calc_z(n_quant, cur_q)
        if (np.array_equal(cur_z, last_z)):
            break
        else:
            last_z = cur_z.copy()

    # build lookup table
    lut = np.zeros(MAXIMUM_PIXEL_INTENSE + 1)
    cur_z = np.ceil(cur_z).astype("uint8")
    for n in range(len(cur_q)):
        lut[cur_z[n]:cur_z[n + 1]] = cur_q[n]

    im_quant = lut[np.ceil(cur_im).astype("uint8")]
    im_quant /= MAXIMUM_PIXEL_INTENSE

    if (is_rgb):
        tmp_im = rgb2yiq(im_orig.copy())
        tmp_im[:, :, 0] = im_quant
        im_quant = yiq2rgb(tmp_im)

    return [im_quant, error]

