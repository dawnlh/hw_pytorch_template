
import torch
import numpy as np
import os
import sys
from tqdm import tqdm

# ===============
# inference time estimation
# ===============


def gpu_inference_time_est(model, inputs_shape, logger=None, device=None, repetitions=100):
    """
    inference time estimation

    Args:
        model: torch model
        inputs_shape ([tuple] | tuple): shape of the model's batch inputs
        logger: logger. Defaults to None
        device: GPU cuda device. Defaults to None, i.e. use model's woring device
        repetitions (int, optional): testing times. Defaults to 100.
    """

    # INIT
    if device is None:
        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
        else:
            raise ValueError("Please assign a GPU device for inference")
    else:
        model.to(device)

    if isinstance(inputs_shape, list):
        dummy_input = [torch.randn(shape_k, dtype=torch.float).to(
            device) for shape_k in inputs_shape]
    elif isinstance(inputs_shape, tuple):
        dummy_input = [torch.randn(inputs_shape, dtype=torch.float).to(
            device)]
    else:
        raise ValueError(
            f"`inputs_shape` should be a tuple or a list containing multiple tuples, but get `{inputs_shape}` ")

    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    #GPU-WARM-UP
    for _ in range(10):
        _ = model(*dummy_input)

    # MEASURE PERFORMANCE
    print("===> Start Inference Time Estimation")
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(*dummy_input)
            ender.record()

            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)
    if logger is not None:
        logger.info('='*20+'\n\tInference \tTime Estimation: \nMean \t {mean_time:.3f}ms, Std \t {std_time:.3f}ms\n'.format(
            mean_time=mean_time, std_time=std_time)+'='*20)
    else:
        print('='*20+'\nInference Time Estimation: \nMean {mean_time:.3f}ms, Std {std_time:.3f}ms\n'.format(
            mean_time=mean_time, std_time=std_time)+'='*20)


if __name__ == '__main__':

    model = DeepRFT()
    device = torch.device("cuda:0")
    inputs_shape = (1, 3, 224, 224)
    inference_time_est(model, device, inputs_shape)
