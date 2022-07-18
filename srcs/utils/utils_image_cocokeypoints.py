import numpy as np
import cv2
import math
import os
from os.path import join as opj


def GuassBlur(img, kernel):
    """
    add gaussian blur
    :param im: BGR image input by opencv
    :param kernel: 3-10
    :method:
    :return modified image
    """
    img = cv2.blur(img, (kernel, kernel))
    return img


def BoxBlur(img, kernel):
    """
    add box blur
    """
    kernel = np.ones((kernel, kernel), np.float32)/(kernel*kernel)
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def MedianBlur(img, odd_kernel):
    """
    add median blur
    """
    dst = cv2.medianBlur(img, odd_kernel)
    return dst


def BilateralBlur(img, diameter):
    """
    rft http://people.csail.mit.edu/sparis/bf_course/
    add bilateral blur
    """
    dst = cv2.bilateralFilter(img, diameter, 75, 75)
    return dst


def MotionBlur(img, kernel):
    """
    add mothon blur
    """
    kernel_motion_blur = np.zeros((kernel, kernel))
    kernel_motion_blur[int((kernel-1)/2), :] = np.ones(kernel)
    kernel_motion_blur = kernel_motion_blur / kernel
    dst = cv2.filter2D(img, -1, kernel_motion_blur)
    return dst


def create_random_trajectory(trajectory_size=64, anxiety=0.005, num_samples=2000, max_total_length=64):
    """
    Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012].
    Each trajectory consists of a complex-valued vector determining the discrete positions of a particle following a
    2-D random motion in continuous domain. The particle has an initial velocity vector which, at each iteration, is
    affected by a Gaussian perturbation and by a deterministic inertial component, directed toward the previous
    particle position. In addition, with a small probability, an impulsive (abrupt) perturbation aiming at inverting
    the particle velocity may arises, mimicking a sudden movement that occurs when the user presses the camera
    button or tries to compensate the camera shake. At each step, the velocity is normalized to guarantee that
    trajectories corresponding to equal exposures have the same length. Each perturbation (Gaussian, inertial, and
    impulsive) is ruled by its own parameter. Rectilinear Blur as in [Boracchi and Foi 2011] can be obtained by
    setting anxiety to 0 (when no impulsive changes occurs)

    :param trajectory_size: size (in pixels) of the square support of the Trajectory curve
    :param anxiety: amount of shake, which scales random vector added at each sample
    :param num_samples: number of samples where the Trajectory is sampled
    :param max_total_length: maximum trajectory length computed as sum of all distanced between consecutive points
    
    Modified: Zhihong Zhang
    Reference: [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from Motion Blur"
    """

    abruptShakesCounter = 0
    totalLength = 0
    # term determining, at each sample, the strength of the component leading towards the previous position
    centripetal = 0.7 * np.random.rand()

    # term determining, at each sample, the random component of the new direction
    gaussianTerm = 10 * np.random.rand()

    # probability of having a big shake, e.g. due to pressing camera button or abrupt hand movements
    freqBigShakes = 0.2 * np.random.rand()

    # v is the initial velocity vector, initialized at random direction
    init_angle = 2*np.pi * np.random.rand()

    # initial velocity vector having norm 1
    v0 = np.cos(init_angle) + 1j * np.sin(init_angle)

    # the speed of the initial velocity vector
    v = v0 * max_total_length/(num_samples-1)

    if anxiety > 0:
        v = v0 * anxiety
    # initialize the trajectory vector
    x = np.zeros(num_samples, dtype=np.complex)

    for t in range(num_samples-1):
        # determine if there is an abrupt (impulsive) shake
        if np.random.rand() < freqBigShakes * anxiety:
            # if yes, determine the next direction which is likely to be opposite to the previous one
            nextDirection = 2 * v * \
                (np.exp(1j * (np.pi + (np.random.rand() - 0.5))))
            abruptShakesCounter = abruptShakesCounter + 1
        else:
            nextDirection = 0

        # determine the random component motion vector at the next step
        dv = nextDirection + anxiety * (gaussianTerm * (np.random.randn(
        ) + 1j * np.random.randn()) - centripetal * x[t]) * (max_total_length / (num_samples - 1))
        v = v + dv

        # velocity vector normalization
        v = (v / np.abs(v)) * max_total_length / (num_samples - 1)
        # update particle position
        x[t + 1] = x[t] + v

        # compute total length
        totalLength = totalLength + np.abs(x[t + 1] - x[t])

    # Center the Trajectory

    # Set the lowest position in zero
    x = x - 1j * np.min(np.imag(x)) - np.min(np.real(x))

    # Center the Trajectory
    x = x - 1j * \
        np.remainder(np.imag(x[0]), 1) - \
        np.remainder(np.real(x[0]), 1) + 1 + 1j
    x = x + 1j * np.ceil((trajectory_size - np.max(np.imag(x))) / 2) + \
        np.ceil((trajectory_size - np.max(np.real(x))) / 2)
    return x, totalLength, abruptShakesCounter


def create_random_psf(psf_size=64, trajSize=64, anxiety=0.005, num_samples=2000, max_total_length=64, exp_time=[1]):
    """
    PSFs are obtained by sampling the continuous trajectory TrajCurve on a regular pixel grid using linear interpolation at subpixel level

    Args:
        % trajectory_size: size (in pixels) of the square support of the trajectory curve
        % anxiety: amount of shake, which scales random vector added at each sample
        % num_samples: number of samples where the Trajectory is sampled
        % max_total_length: maximum trajectory length computed as sum of all distanced between consecutive points
        % psf_size   Size of the PFS where the TrajCurve is sampled
        % exp_time         Vector of exposure times: for each of them a PSF will be generated, default = [1]

    Returns:
        blur kernel as NumPy array of shape [PSFsize, PSFsize]
        
    Modified: Zhihong Zhang
    Reference: [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from Motion Blur"
    """

    x, _, _ = create_random_trajectory(
        trajSize, anxiety, num_samples, max_total_length)
    psf_size = (psf_size, psf_size)

    if isinstance(exp_time, (int, float)):
        exp_time = [exp_time]

    # PSFnumber = len(exp_time)
    numt = len(x)

    # center with respect to baricenter
    x = x - np.mean(x) + (psf_size[1] + 1j * psf_size[0] + 1 + 1j) / 2

    #    x = np.max(1, np.min(PSFsize[1], np.real(x))) + 1j*np.max(1, np.min(PSFsize[0], np.imag(x)))

    # generate PSFs
    PSFs = []
    PSF = np.zeros(psf_size)

    def triangle_fun(d):
        return max(0, (1 - np.abs(d)))

    def triangle_fun_prod(d1, d2):
        return triangle_fun(d1) * triangle_fun(d2)

    # set the exposure time
    for jj in range(len(exp_time)):
        try_times = 0
        while(try_times < 200):
            try_times = try_times+1
            if jj == 0:
                prevT = 0
            else:
                prevT = exp_time[jj - 1]

            # sample the trajectory until time exp_time
            for t in range(len(x)):
                if (exp_time[jj] * numt >= t) and (prevT * numt < t - 1):
                    t_proportion = 1
                elif (exp_time[jj] * numt >= t - 1) and (prevT * numt < t - 1):
                    t_proportion = exp_time[jj] * numt - t + 1
                elif (exp_time[jj] * numt >= t) and (prevT * numt < t):
                    t_proportion = t - prevT * numt
                elif (exp_time[jj] * numt >= t - 1) and (prevT * numt < t):
                    t_proportion = (exp_time[jj] - prevT) * numt
                else:
                    t_proportion = 0

                m2 = min(psf_size[1] - 2, max(1, int(np.floor(np.real(x[t])))))
                M2 = m2 + 1
                m1 = min(psf_size[0] - 2, max(1, int(np.floor(np.imag(x[t])))))
                M1 = m1 + 1

                # linear interp. (separable)
                PSF[m1, m2] = PSF[m1, m2] + t_proportion * \
                    triangle_fun_prod(np.real(x[t]) - m2, np.imag(x[t]) - m1)
                PSF[m1, M2] = PSF[m1, M2] + t_proportion * \
                    triangle_fun_prod(np.real(x[t]) - M2, np.imag(x[t]) - m1)
                PSF[M1, m2] = PSF[M1, m2] + t_proportion * \
                    triangle_fun_prod(np.real(x[t]) - m2, np.imag(x[t]) - M1)
                PSF[M1, M2] = PSF[M1, M2] + t_proportion * \
                    triangle_fun_prod(np.real(x[t]) - M2, np.imag(x[t]) - M1)
            if(not np.any(np.isnan(PSF))):
                break

        PSFs.append(PSF / PSF.sum())
        if len(PSFs) == 1:
            PSFs = PSFs[0]
    return PSFs


def RandomMotionBlur(img, psf_size=64):
    """
    add random mothon blur
    rft [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from Motion Blur"
    """
    kernel_motion_blur = create_random_psf(psf_size)
    dst = cv2.filter2D(img, -1, kernel_motion_blur)
    return dst


def AugImage(img):
    """
    Aug image randomly
    """
    Dices = np.random.randn(4,)
    BlurDice = np.random.randn()
    if Dices[0] > 0.8:
        img = Contrast(img)
    if Dices[1] > 0.8:
        img = Bright(img)
    if Dices[2] > 0.8:
        img = Saturation(img)
    if Dices[3] > 0.8:
        kerSize = np.int(np.random.randn()*5)
        kerSize = np.max([kerSize, 3])
        if BlurDice < 0.25:
            img = BoxBlur(img, 3)
        elif BlurDice < 0.5:
            kerSize = kerSize/2*2 + 1
            img = MedianBlur(img, kerSize)
        elif BlurDice < 0.75:
            kerSize = kerSize/2*2 + 1
            img = MotionBlur(img, kerSize)
    return img


def GrayWorld(img):
    """
    White balance algorithm
    """
    [h, w, c] = img.shape
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    K = 127.0
    R, G, B = cv2.split(RGB)
    Aver_R = np.average(np.average(R))
    Aver_G = np.average(np.average(G))
    Aver_B = np.average(np.average(B))
    #K =    (Aver_B +   Aver_G +   Aver_R)/3.0
    K_R = K/Aver_R
    K_G = K/Aver_G
    K_B = K/Aver_B
    R = R * K_R
    G = G * K_G
    B = B * K_B
    R = np.where(R > 255, 255, R)
    R = np.where(R < 0, 0, R)
    G = np.where(G > 255, 255, G)
    G = np.where(G < 0, 0, G)
    B = np.where(B > 255, 255, B)
    B = np.where(B < 0, 0, B)
    RGB[:, :, 0] = np.uint8(R)
    RGB[:, :, 1] = np.uint8(G)
    RGB[:, :, 2] = np.uint8(B)
    BGR = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)
    return BGR


def resize(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale,
                    fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale


def transform(im, pixel_means, scale=1.0):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    if len(im.shape) == 3 and im.shape[2] == 3:
        im_tensor = np.zeros(
            (1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = (
                im[:, :, 2 - i] - pixel_means[2 - i]) * scale
    elif len(im.shape) == 2:
        im_tensor = np.zeros(
            (1, 1, im.shape[0], im.shape[1]), dtype=np.float32)
        im_tensor[0, 0, :, :] = (im[:, :] - pixel_means[0]) * scale
    else:
        raise ValueError("can not transform image successfully")
    return im_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice,
                       :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1],
                       :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor


def addMaskImage(img):
    """
    add random mask on image by some pure image patch 
    to be improved
    :param im: BGR image input by opencv
    """
    [h, w, c] = img.shape
    h_start = np.random.randint(h/2, h-1)
    w_start = np.random.randint(w/2, w-1)
    img[h_start:h-1, :, 0] = np.random.randint(0, 120)
    img[h_start:h-1, :, 1] = np.random.randint(0, 120)
    img[h_start:h-1, :, 2] = np.random.randint(0, 120)
    img[:, w_start:w-1, 0] = np.random.randint(0, 120)
    img[:, w_start:w-1, 1] = np.random.randint(0, 120)
    img[:, w_start:w-1, 2] = np.random.randint(0, 120)
    img = np.uint8(img)
    return img, h_start, w_start


def Contrast(img):
    """
    adjust the contrast of the image,  w.r.t
    http://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    :param im: BGR image input by opencv
    :param factor: constract factor, [-128, 128]
    :method: fvalue = 259.0/255.0 * (factor + 255.0)/(259.0-factor)
             pixel = (pixel - 128.0) * fvalue + 128.0
    :return modified image
    """
    factor = 2 * (np.random.rand() - 0.5) * 128
    assert (factor <= 128 and factor >= -128), 'contract factor value wrong'
    fvalue = 259.0/255.0 * (factor + 255.0)/(259.0-factor)
    img = np.round((img - 128.0)*fvalue + 128.0)
    img = np.where(img > 255, 255, img)
    img = np.where(img < 0, 0, img)
    img = np.uint8(img)
    return img


def Bright(img):
    """
    adjust the brightneess of the image
    :param im: BGR image input by opencv
    :param factor: brightness factor, [0, 2], <1 dark, >1 bright
    :method:
    :return modified image
    """
    factor = 2 * np.random.rand()
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    V = V * np.float(factor)
    V = np.where(V > 255, 255, V)
    V = np.where(V < 0, 0, V)
    HSV[:, :, 2] = np.uint8(V)
    BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    return BGR


def Saturation(img):
    """
    adjust the saturation of the image
    :param im: BGR image input by opencv
    :param factor: brightness factor, [0, 2]
    :method:
    :return modified image
    """
    factor = 2 * np.random.rand()
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    S = S * np.float(factor)
    S = np.where(S > 255, 255, S)
    S = np.where(S < 0, 0, S)
    HSV[:, :, 1] = np.uint8(S)
    BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    return BGR


if __name__ == '__main__':
    # image = cv2.imread('test11.jpg')
    # im_vis = image.copy()
    # image_blur = GrayWorld(image)
    # cv2.imshow('blur', image_blur)
    # cv2.waitKey(0)
    # cv2.imshow('test', im_vis)
    # cv2.waitKey(0)

    save_dir = './outputs/tmp/test/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for k in range(100):
        max_total_length = np.random.uniform(15, 40)
        anxiety = np.random.uniform(0.004, 0.006)
        num_samples = np.int(np.random.uniform(1500, 2000))
        psf = create_random_psf(psf_size=64, trajSize=64, anxiety=anxiety,
                                num_samples=num_samples, max_total_length=max_total_length, exp_time=[1])
        psf = psf/np.max(psf)*255
        print("PSF_%d" % k)
        cv2.imwrite(opj(save_dir, 'psf%02d.png' % k),
                    psf, [cv2.IMWRITE_PNG_COMPRESSION, 0])
