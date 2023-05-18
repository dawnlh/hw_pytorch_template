

def multiCenterInterpOrder(index1, index2):
    """
    determine the interpolatation order in multi-frame central interpolation task
    It's a recursive problem.
    Args:
        index1: frame index of the prior frame
        index2: frame index of the posterior frame
    Returns: [[existing_index1, existing_index2, generated_new_index], ...
    Note:
        task definition: progressively generate multiple inner frames from two given frames using an interpolation algorithm which can only generate the center frame between two adjacent frames
        e.g. index1=0 and index2=4, the resulting interpolation order is: [0, 4->2], [0, 2->1], [2, 4->3]

    """
    # interpolation factor, i.e. frame rate upscale time, e.g., generate 7 frames = 8x interpolation
    interp_fc = index2 - index1

    assert (interp_fc & (interp_fc-1) ==
            0) and interp_fc != 0, 'param. $interp_fc should be 2^n'
    x = []

    def recur_interp(start_index, end_index):
        if end_index-start_index == 1:
            return
        assert (start_index +
                end_index) % 2 == 0, 'start_index + end_index should be even'
        mid_index = (start_index + end_index) // 2
        x.append([start_index, end_index, mid_index])
        recur_interp(start_index, mid_index)
        recur_interp(mid_index, end_index)

    recur_interp(index1, index2)
    return x
