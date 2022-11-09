

import math

# ==============================
# reception field calculation
# by: https://rubikscode.net/2021/11/15/receptive-field-arithmetic-for-convolutional-neural-networks/
# more:
#   https://github.com/Fangyh09/pytorch-receptive-field
#   https://github.com/Fangyh09/pytorch-receptive-field
#==============================


class ReceptiveFieldCalculator():
    #Assume the two dimensions are the same
    #Each kernel requires the following parameters:
    # - k_i: kernel size
    # - s_i: stride
    # - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
    #
    #Each layer i requires the following parameters to be fully represented:
    # - n_i: number of feature (data layer has n_1 = imagesize )
    # - r_i: receptive field of a feature in layer i
    # - j_i: distance (projected to image pixel distance) between center of two adjacent features
    # - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

    def calculate(self, architecture, input_image_size):
        """
        architecture: network arch info dict {layer_name:[kernel_size, stride, padding]}, e.g. {'conv1': [11, 4, 0],'pool1': [3, 2, 0],...}
        """
        input_layer = ('input_layer', input_image_size, 1, 1, 0.5)
        self._print_layer_info(input_layer)

        for key in architecture:
            current_layer = self._calculate_layer_info(
                architecture[key], input_layer, key)
            self._print_layer_info(current_layer)
            input_layer = current_layer

    def _print_layer_info(self, layer):
        print(f'------')
        print(
            f'{layer[0]}: feature_size = {layer[1]}; receptive field = {layer[3]}; jump = {layer[2]}; start = {layer[4]}')
        print(f'------')

    def _calculate_layer_info(self, current_layer, input_layer, layer_name):
        n_in = input_layer[1]
        j_in = input_layer[2]
        r_in = input_layer[3]
        start_in = input_layer[4]

        k = current_layer[0]
        s = current_layer[1]
        p = current_layer[2]

        n_out = math.floor((n_in - k + 2*p)/s) + 1
        padding = (n_out-1)*s - n_in + k
        p_right = math.ceil(padding/2)
        p_left = math.floor(padding/2)

        j_out = j_in * s
        r_out = r_in + (k - 1)*j_in
        start_out = start_in + ((k-1)/2 - p_left)*j_in
        return layer_name, n_out, j_out, r_out, start_out


# =============
# main function
# =============
if __name__ == "__main__":
    # ---------------------------
    # reception field calculation
    # --------------------------
    alex_net = {
        'conv1': [11, 4, 0],
        'pool1': [3, 2, 0],
        'conv2': [5, 1, 2],
        'pool2': [3, 2, 0],
        'conv3': [3, 1, 1],
        'conv4': [3, 1, 1],
        'conv5': [3, 1, 1],
        'pool5': [3, 2, 0],
        'fc6-conv': [6, 1, 0],
        'fc7-conv': [1, 1, 0]
    }
    calculator = ReceptiveFieldCalculator()
    calculator.calculate(alex_net, 227)
