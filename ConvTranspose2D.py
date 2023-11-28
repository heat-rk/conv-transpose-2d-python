import numpy as np


class ConvTranspose2D:
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            dilation=1,
            bias=True,
            padding_mode='zeros',
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)

        if isinstance(dilation, tuple):
            self.dilation = dilation
        else:
            self.dilation = (dilation, dilation)

        if isinstance(stride, tuple):
            self.stride = stride
        else:
            self.stride = (stride, stride)

        if isinstance(padding, tuple):
            self.pad = padding
        elif padding == 'same':
            if self.stride[0] != 1 or self.stride[1] != 1:
                raise ValueError('padding == \'same\' can be applied only with stride = 1')
            self.pad = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        elif padding == 'valid':
            self.pad = (0, 0)
        else:
            self.pad = (padding, padding)

        if isinstance(output_padding, tuple):
            self.out_pad = output_padding
        else:
            self.out_pad = (output_padding, output_padding)

        if self.out_pad[0] >= self.stride[0] and self.out_pad[1] >= self.stride[1] and \
                self.out_pad[0] >= self.dilation[0] and self.out_pad[1] >= self.dilation[1]:
            raise ValueError('Output padding must be smaller than either stride or dilation')

        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for ConvTranspose2D')

        self.pad_mode = padding_mode
        self.bias = bias

    def forward(self, input, weight, bias):
        """
        :param input: array of input images of format NCHW
        :param weight: array of learning weights of format DCHW
        :param bias: array of biases
        :return: output images of format NDHW
        """

        batches = len(input)
        out = []

        for b in range(batches):
            strided = self.__stride_input(input[b], self.stride)
            h_in = strided.shape[1]
            w_in = strided.shape[2]

            h_out = int(
                (h_in - 1) - 2 * self.pad[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.out_pad[0] + 1)

            w_out = int(
                (w_in - 1) - 2 * self.pad[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.out_pad[1] + 1)

            out.append(np.zeros((self.out_channels, h_out, w_out)))

            for c_out in range(self.out_channels):
                for y_out in range(h_out):
                    for x_out in range(w_out):
                        sum = 0
                        for c_in in range(self.in_channels):
                            for kernel_y in range(self.kernel_size[0]):
                                for kernel_x in range(self.kernel_size[1]):
                                    y_in = y_out + kernel_y * self.dilation[0] - (
                                                self.dilation[0] * (self.kernel_size[0] - 1) - self.pad[0])
                                    x_in = x_out + kernel_x * self.dilation[1] - (
                                                self.dilation[1] * (self.kernel_size[1] - 1) - self.pad[1])
                                    if 0 <= y_in < h_in and 0 <= x_in < w_in:
                                        sum += strided[c_in][y_in][x_in] * weight[c_out][c_in][kernel_y][kernel_x]

                        out[b][c_out][y_out][x_out] = sum + (bias[c_out] if self.bias else 0)

        return np.array(out)

    @staticmethod
    def __stride_input(input, stride):
        channels, rows, cols = input.shape
        out_rows, out_cols = rows * stride[0] - stride[0] + 1, cols * stride[1] - stride[1] + 1
        out = np.zeros((channels, out_rows, out_cols), input.dtype)

        for c in range(0, channels):
            for y in range(0, out_rows, stride[0]):
                for x in range(0, out_cols, stride[1]):
                    out[c, y, x] = input[c, int(y / stride[0]), int(x / stride[1])]

        return out
