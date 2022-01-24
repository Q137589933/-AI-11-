import paddle.nn as nn
import numpy as np
import paddle
class L1_Advanced_Sobel_Loss(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv_op_x = nn.Conv2D(3,3, 3,bias_attr=False)
        self.conv_op_y = nn.Conv2D(3,3, 3,bias_attr=False)

        sobel_kernel_x = np.array([[[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]]], dtype='float32')
        sobel_kernel_y = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = paddle.to_tensor(sobel_kernel_x)
        self.conv_op_y.weight.data = paddle.to_tensor(sobel_kernel_y)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    # def forward(self, edge_outputs, image_target):
    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = paddle.abs(edge_Y_xoutputs) + paddle.abs(edge_Y_youtputs)

        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = paddle.abs(edge_Y_x) + paddle.abs(edge_Y_y)

        diff = paddle.add(edge_Youtputs, -edge_Y)
        error = paddle.abs(diff)
        loss = paddle.mean(error)# / outputs.size(0)
        return loss