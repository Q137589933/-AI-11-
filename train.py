import argparse

import sys
from copy import copy

from numpy import mean

from LossNet import L1_Advanced_Sobel_Loss
from dataset import *
from log import Logger
from model import *
from paddle.metric import Metric
import paddle
import numpy as np
import math
import warnings
import datetime
from utils import load_pretrained_model, psnr

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='data/data121607/data/train')

    parser.add_argument(
        '--valset_root',
        dest='valset_root',
        help='The path of valset root',
        type=str,
        default=None)

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=32
    )
    parser.add_argument(
        '--start_epoch',
        dest='start_epoch',
        help='start_epoch',
        type=int,
        default=0
    )
    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=100
    )
    parser.add_argument(
        '--down_lr_epochs',
        dest='down_lr_epochs',
        help='after how many epochs,divide your learning rate by 5',
        type=int,
        default=10
    )
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='define learning rate',
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=10
    )

    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='save_interval',
        type=int,
        default=1
    )

    parser.add_argument(
        '--sample_interval',
        dest='sample_interval',
        help='sample_interval',
        type=int,
        default=100
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=1234
    )
    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)
    parser.add_argument(
        '--weight_decay',
        dest='weight_decay',
        help='weight_decay',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--use_elastic',
        dest='use_elastic',
        help='use_elastic',
        action='store_true'
    )
    parser.add_argument(
        '--crop_center',
        dest='crop_center',
        help='crop_center',
        action='store_true'
    )
    parser.add_argument(
        '--use_mosaic',
        dest='use_mosaic',
        help='use_mosaic or not',
        action='store_true',
    )
    return parser.parse_args()


def sample_images(epoch, i, real_A, real_B, fake_B):
    data, pred, label = real_A * 255, fake_B * 255, real_B * 255
    pred = paddle.clip(pred.detach(), 0, 255)
    data = data.cast('int64')
    pred = pred.cast('int64')
    label = label.cast('int64')
    h, w = pred.shape[-2], pred.shape[-1]
    img = np.zeros((h, 1 * 3 * w, 3))
    for idx in range(0, 1):
        row = idx * h
        tmplist = [data[idx], pred[idx], label[idx]]
        for k in range(3):
            col = k * w
            tmp = np.transpose(tmplist[k], (1, 2, 0))
            img[row:row + h, col:col + w] = np.array(tmp)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    if not os.path.exists("./train_result"):
        os.makedirs("./train_result")
    img.save("./train_result/%03d_%06d.png" % (epoch, i))


def main(args):
    paddle.disable_static()
    sys.stdout = Logger("./work/train.log")
    train_dataset = ImageDataset(root=args.dataset_root, training=True, val=False, use_elastic=args.use_elastic,
                                 crop_center=args.crop_center)
    valid_dataset = ImageDataset(root=args.valset_root, training=True, val=True)

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=1)
    val_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=1,
                            num_workers=1)
    print('loaded dataset successfully!')
    print(f'the number of training set images: {train_dataset.__len__()}')

    # 预计模型结构生成模型对象，便于进行后续的配置、训练和验证
    multi_output = True
    model = MBCNN(64, multi_output)  # MBCNN-light: model = MBCNN(32,multi_output)
    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)
    optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate, parameters=model.parameters(),
                                      beta1=0.5, beta2=0.999, weight_decay=args.weight_decay)
    # 设置损失函数
    loss_fn = paddle.nn.L1Loss(reduction='mean')
    loss_asl = L1_Advanced_Sobel_Loss()
    val_loss_list = []  # 记录验证机loss
    prev_time = time.time()
    coef = 0.25
    min_val_loss = 1e9
    min_val_loss_epoch = args.start_epoch
    for epoch in range(args.start_epoch + 1, args.max_epochs + 1):
        model.train()
        # 连续5个epoch loss不降或降幅过小，学习率减半
        # if epoch - min_val_loss_epoch >= 10:
        #     optimizer.set_lr(optimizer.get_lr() / 2)
        if (epoch - args.start_epoch) % args.down_lr_epochs == 0:
            optimizer.set_lr(optimizer.get_lr() / 2)

        for batch_id, data in enumerate(train_loader):
            x_data = data[0]  # 训练数据
            y_data_list = data[1]  # 训练数据标签
            # x_data=paddle.ones(shape=[2, 3,256,256])
            # y_data=paddle.ones(shape=[1, 128,128,128])
            z3, z2, z1 = model(x_data)  # 预测结果
            y_data_1, y_data_2, y_data_3 = y_data_list
            loss1 = loss_fn(z1, y_data_1) + coef * loss_asl(z1, y_data_1)
            loss2 = loss_fn(z2, y_data_2) + coef * loss_asl(z2, y_data_2)
            loss3 = loss_fn(z3, y_data_3) + coef * loss_asl(z3, y_data_3)
            # loss4 = loss_fn(z4, y_data_4) + coef * loss_asl(z4, y_data_4)
            loss = loss1 + loss2 + loss3
            # 计算损失 等价于 prepare 中loss的设置
            loss.backward()
            optimizer.step()
            # 梯度清零
            optimizer.clear_grad()
            model.clear_gradients()
            batches_done = epoch * len(train_loader) + batch_id
            batches_left = args.max_epochs * len(train_loader) - batches_done

            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            if batch_id % args.log_iters == 0:
                print('')
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [loss1: %f] [loss2: %f] [loss3: %.7f] [lr: %f] ETA: %s" %
                    (epoch, args.max_epochs,
                     batch_id, len(train_loader),
                     loss.numpy()[0],
                     loss1.numpy()[0],
                     loss2.numpy()[0],
                     loss3.numpy()[0],
                     # loss4.numpy()[0],
                     optimizer.get_lr(),
                     time_left))
            if batch_id % args.sample_interval == 0:
                sample_images(epoch, batch_id, x_data, y_data_1, z1)
        if epoch % args.save_interval == 0:
            current_save_dir = os.path.join("train_result", "model", f'epoch_{epoch}')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
            if val_loader is not None:
                loss_list = []
                loss_1_list = []
                PSNR_list = []
                for id, val_batch in enumerate(val_loader):
                    model.eval()
                    x_data = val_batch[0]
                    y_data_list = val_batch[1]
                    z3, z2, z1 = model(x_data)  # 预测结果
                    y_data_1, y_data_2, y_data_3 = y_data_list
                    loss1 = loss_fn(z1, y_data_1) + coef * loss_asl(z1, y_data_1)
                    loss2 = loss_fn(z2, y_data_2) + coef * loss_asl(z2, y_data_2)
                    loss3 = loss_fn(z3, y_data_3) + coef * loss_asl(z3, y_data_3)
                    # loss4 = loss_fn(z4, y_data_4) + coef * loss_asl(z4, y_data_4)
                    loss = loss1 + loss2 + loss3
                    loss_list.append(loss.numpy()[0])
                    loss_1_list.append(loss1.numpy()[0])
                    model.clear_gradients()
                    PSNR = psnr(y_data_1, z1)
                    PSNR_list.append(PSNR)
                    if id % args.sample_interval == 0:
                        sample_images(epoch, id, x_data, y_data_1, z1)
                val_loss = mean(np.array(loss_list), dtype=float).squeeze()
                val_loss_list.append(val_loss)
                if val_loss < min_val_loss:
                    min_val_loss_epoch = epoch
                    min_val_loss = val_loss
                sys.stdout.write(
                    "\r validation loss:[Epoch %d/%d] [mean loss: %f] [mean PSNR: %f]" %
                    (epoch,
                     args.max_epochs,
                     mean(np.array(loss_list), dtype=float).squeeze(),
                     mean(np.array(PSNR_list), dtype=float).squeeze(),
                     ))
            # 更新参数


if __name__ == '__main__':
    args = parse_args()
    main(args)
