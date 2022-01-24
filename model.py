import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from math import cos, pi, sqrt


class depth_to_space(nn.Layer):
    def __init__(self, scale_factor):
        super(depth_to_space, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, tensor):
        num, ch, height, width = tensor.shape
        if ch % (self.scale_factor * self.scale_factor) != 0:
            raise ValueError('channel of tensor must be divisible by '
                             '(scale_factor * scale_factor).')

        new_ch = ch // (self.scale_factor * self.scale_factor)
        new_height = height * self.scale_factor
        new_width = width * self.scale_factor

        tensor = tensor.reshape(
            [num, self.scale_factor, self.scale_factor, new_ch, height, width]).clone()
        # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
        tensor = tensor.transpose([0, 3, 4, 1, 5, 2])
        tensor = tensor.reshape([num, new_ch, new_height, new_width])
        return tensor


class space_to_depth(nn.Layer):
    def __init__(self, scale_factor):
        super(space_to_depth, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, tensor):
        num, ch, height, width = tensor.shape
        if height % self.scale_factor != 0 or width % self.scale_factor != 0:
            raise ValueError('height and widht of tensor must be divisible by '
                             'scale_factor.')

        new_ch = ch * (self.scale_factor * self.scale_factor)
        new_height = height // self.scale_factor
        new_width = width // self.scale_factor

        tensor = tensor.reshape(
            [num, ch, new_height, self.scale_factor, new_width, self.scale_factor]).clone()
        # new axis: [num, scale_factor, scale_factor, ch, new_height, new_width]
        tensor = tensor.transpose([0, 3, 5, 1, 2, 4])
        tensor = tensor.reshape([num, new_ch, new_height, new_width])
        return tensor


# IDCT
class adaptive_implicit_trans(nn.Layer):
    def __init__(self):
        super(adaptive_implicit_trans, self).__init__()
        conv_shape = (1, 1, 64, 64)

        self.it_weights = paddle.create_parameter(
            shape=[1, 1, 64, 1], dtype='float32',
            default_initializer=nn.initializer.Constant(value=1.0))

        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0 / 8)
        r2 = sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = cos(_u * u * pi / 16) * cos(_v * v * pi / 16)
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[0, 0, index2, index] = t
        self.kernel = paddle.to_tensor(kernel, dtype='float32')

    def forward(self, inputs):

        k = (self.kernel * self.it_weights).reshape([64, 64, -1, 1])
        y = nn.functional.conv2d(inputs,
                                 k,
                                 padding='SAME', groups=inputs.shape[1] // 64,
                                 data_format='NCHW')

        return y


class ScaleLayer(nn.Layer):
    def __init__(self, s):
        super(ScaleLayer, self).__init__()
        self.s = s
        self.kernel = paddle.create_parameter(
            shape=[1, ], dtype='float32',
            default_initializer=nn.initializer.Constant(s))

    def forward(self, inputs):
        return inputs * self.kernel


class conv_rl(nn.Layer):
    def __init__(self, inf, filters, kernel, padding='SAME', use_bias=True, dilation_rate=1, strides=(1, 1),
                 use_bn=False):

        super(conv_rl, self).__init__()
        self.filters = filters
        self.kernel = kernel
        self.dilation_rate = dilation_rate
        if dilation_rate == 0:
            self.convr1 = nn.Conv2D(inf, filters, 1, padding=padding)
        else:
            self.convr2 = nn.Conv2D(inf, filters, kernel, padding=padding,
                                    dilation=dilation_rate,
                                    stride=strides)
        self.bn = nn.BatchNorm2D(filters)
        self.relu = nn.ReLU()
        self.use_bn = use_bn

    def forward(self, x):
        if self.dilation_rate == 0:
            y = self.convr1(x)
        else:
            y = self.convr2(x)
        y = self.relu(y)
        if self.use_bn:
            y = self.bn(y)
        return y


class conv(nn.Layer):
    def __init__(self, inf, filters, kernel, padding='SAME', use_bias=True, dilation_rate=1, strides=(1, 1)):
        super(conv, self).__init__()
        self.conv = nn.Conv2D(inf, filters, kernel, padding=padding,
                              dilation=dilation_rate, stride=strides)

    def forward(self, x):
        y = self.conv(x)
        return y


class MTRB(nn.Layer):
    def __init__(self, d_list, nFilters, enbale=True):
        super(MTRB, self).__init__()
        self.d_list = d_list
        self.nFilters = nFilters
        self.enable = True
        layer_list = []
        # dense block
        for i in range(len(self.d_list)):
            j = i + 1
            conv_rl_layer = conv_rl(self.nFilters + j * self.nFilters,
                                    self.nFilters,
                                    3,
                                    dilation_rate=self.d_list[i])
            layer_list.append(conv_rl_layer)
        self.cr_layers = nn.LayerList(layer_list)
        self.conv1 = conv(self.nFilters * 7, 64, 3)
        self.IDCT = adaptive_implicit_trans()
        self.conv2 = conv(self.nFilters, self.nFilters * 2, 1)
        self.FSL = ScaleLayer(s=0.1)
        self.lamb = lambda x: x * 0

    def forward(self, x):
        t = x.detach()
        for layer in self.cr_layers:
            _t = layer(t)
            t = paddle.concat([_t, t], axis=1)

        t = self.conv1(t)

        t = self.IDCT(t)
        t = self.conv2(t)
        t = self.FSL(t)
        if not self.enable:
            t = self.lamb(t)
        t = x + t
        return t


class LTMB(nn.Layer):
    def __init__(self, d_list, nFilters):
        super(LTMB, self).__init__()
        self.d_list = d_list
        self.nFilters = nFilters
        self.enable = True
        layer_list = []
        for i in range(len(self.d_list)):
            j = i + 1
            conv_rl_layer = conv_rl(self.nFilters + j * self.nFilters,
                                    self.nFilters,
                                    3,
                                    dilation_rate=self.d_list[i])
            layer_list.append(conv_rl_layer)
        self.cr_layers1 = nn.LayerList(layer_list)
        self.cr2 = conv_rl(self.nFilters * 7, self.nFilters * 2, 1)

    def forward(self, x):
        t = x
        for layer in self.cr_layers1:
            _t = layer(t)
            t = paddle.concat([_t, t], axis=1)
        t = self.cr2(t)
        return t


class GTMB(nn.Layer):
    def __init__(self, nFilters):
        super(GTMB, self).__init__()
        self.nFilters = nFilters
        self.pad = nn.Pad2D(padding=1)
        self.cr0 = conv_rl(nFilters * 2, nFilters * 4, 3, strides=(2, 2))
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Linear(nFilters * 4, nFilters * 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nFilters * 16, nFilters * 8)
        self.fc3 = nn.Linear(nFilters * 8, nFilters * 4)
        self.cr1 = conv_rl(nFilters * 2, nFilters * 4, 1)
        self.cr2 = conv_rl(nFilters * 4, nFilters * 2, 1)

    def forward(self, x):
        t = self.pad(x)
        t = self.cr0(t)
        t = self.gap(t)
        t = paddle.squeeze(t, axis=[-2, -1])
        t = self.fc1(t)
        t = self.relu(t)
        t = self.fc2(t)
        t = self.relu(t)
        t = self.fc3(t)

        _t = self.cr1(x)
        t = t.unsqueeze(-1).unsqueeze(-1)
        _t = paddle.multiply(_t, t)
        _t = self.cr2(_t)
        return _t


class MBCNN(nn.Layer):
    def __init__(self, nFilters, multi=True):
        super(MBCNN, self).__init__()
        d_list_a = (1, 2, 3, 2, 1)
        d_list_b = (1, 2, 3, 2, 1)
        d_list_c = (1, 2, 2, 2, 1)
        # d_list_d = (1, 2, 2, 2, 1)
        self.nFilters = nFilters
        self.multi = multi
        self.pad = nn.Pad2D(padding=1)
        # branch 1
        self.std = space_to_depth(scale_factor=2)  # 下采样
        self.b1_conv_rl_k3 = conv_rl(12, self.nFilters * 2, kernel=3, padding='SAME')
        self.b1_MTRB_1 = MTRB(d_list_a, self.nFilters, True)
        self.b1_conv_rl_k1 = conv_rl(self.nFilters * 2 + 3, self.nFilters * 2, kernel=1)
        self.b1_GTMB_1 = GTMB(self.nFilters)
        self.b1_MTRB_2 = MTRB(d_list_a, self.nFilters, True)
        self.b1_GTMB_2 = GTMB(self.nFilters)
        self.b1_LTMB = LTMB(d_list_a, self.nFilters)
        self.b1_out_conv = conv(self.nFilters * 2, 12, kernel=3)
        self.b1_dts = depth_to_space(scale_factor=2)
        # branch 2
        self.b2_conv_rl_k3 = conv_rl(self.nFilters * 2, self.nFilters * 2, 3, padding='VALID', strides=(2, 2))
        self.b2_MTRB_1 = MTRB(d_list_b, self.nFilters, True)
        self.b2_conv_rl_k1 = conv_rl(self.nFilters * 2 + 3, self.nFilters * 2, 1)
        self.b2_GTMB_1 = GTMB(self.nFilters)
        self.b2_MTRB_2 = MTRB(d_list_b, self.nFilters, True)
        self.b2_GTMB_2 = GTMB(self.nFilters)
        self.b2_LTMB = LTMB(d_list_b, self.nFilters)
        self.b2_out_conv = conv(self.nFilters * 2, 12, 3)
        self.b2_dts = depth_to_space(scale_factor=2)
        # branch 3
        self.b3_conv_rl_k3 = conv_rl(self.nFilters * 2, self.nFilters * 2, 3, padding='VALID', strides=(2, 2))
        self.b3_MTRB = MTRB(d_list_c, self.nFilters, True)
        # self.b3_conv_rl_k1 = conv_rl(self.nFilters * 2 + 3, self.nFilters * 2, 1)
        self.b3_GTMB = GTMB(self.nFilters)
        # self.b3_MTRB_2 = MTRB(d_list_c, self.nFilters, True)
        # self.b3_GTMB_2 = GTMB(self.nFilters)
        self.b3_LTMB = LTMB(d_list_c, self.nFilters)
        self.b3_out_conv = conv(self.nFilters * 2, 12, 3)
        self.b3_dts = depth_to_space(scale_factor=2)
        # branch 4
        # self.b4_conv_rl_k3 = conv_rl(self.nFilters * 2, self.nFilters * 2, 3, padding='VALID', strides=(2, 2))
        # self.b4_MTRB = MTRB(d_list_d, self.nFilters, True)
        # self.b4_GTMB = GTMB(self.nFilters)
        # self.b4_LTMB = LTMB(d_list_d, self.nFilters)
        # self.b4_out_conv = conv(self.nFilters * 2, 12, 3)
        # self.b4_dts = depth_to_space(scale_factor=2)

    def forward(self, x):
        output_list = []
        # branch 1
        x = self.std(x)
        x = self.b1_conv_rl_k3(x)  # 8m*8m
        x = self.b1_MTRB_1(x)

        # branch 2
        x2 = self.pad(x)
        x2 = self.b2_conv_rl_k3(x2)  # 4m*4m
        x2 = self.b2_MTRB_1(x2)

        # branch 3
        x3 = self.pad(x2)
        x3 = self.b3_conv_rl_k3(x3)  # 2m*2m
        x3 = self.b3_MTRB(x3)

        # branch 4
        # x4 = self.pad(x3)
        # x4 = self.b4_conv_rl_k3(x4)
        # x4 = self.b4_MTRB(x4)
        # x4 = self.b4_GTMB(x4)
        # x4 = self.b4_LTMB(x4)
        # x4 = self.b4_out_conv(x4)
        # x4 = self.b4_dts(x4)
        # output_list.append(x4)
        # branch 3
        # x3 = paddle.concat([x4, x3], axis=1)
        # x3 = self.b3_conv_rl_k1(x3)
        x3 = self.b3_GTMB(x3)
        # x3 = self.b3_MTRB_2(x3)
        # x3 = self.b3_GTMB_2(x3)
        x3 = self.b3_LTMB(x3)
        x3 = self.b3_out_conv(x3)
        x3 = self.b3_dts(x3)  # 4m*4m
        output_list.append(x3)
        # branch 2
        x2 = paddle.concat([x3, x2], axis=1)
        x2 = self.b2_conv_rl_k1(x2)
        x2 = self.b2_GTMB_1(x2)
        x2 = self.b2_MTRB_2(x2)
        x2 = self.b2_GTMB_2(x2)
        x2 = self.b2_LTMB(x2)
        x2 = self.b2_out_conv(x2)
        x2 = self.b2_dts(x2)  # 8m*8m
        output_list.append(x2)
        # branch 1
        x = paddle.concat([x,x2], axis=1)
        x = self.b1_conv_rl_k1(x)
        x = self.b1_GTMB_1(x)
        x = self.b1_MTRB_2(x)
        x = self.b1_GTMB_2(x)

        x = self.b1_LTMB(x)
        x = self.b1_out_conv(x)
        out = self.b1_dts(x)  # 16m*16m
        output_list.append(out)

        if self.multi != True:
            return out
        else:
            return output_list
