# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" PointNet segmentation training script."""

import argparse
import os
import logging
import numpy as np
import mindspore as ms
import datetime
import mindspore
from tqdm import tqdm
from pathlib import Path
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import save_checkpoint
from mindspore.train import Model
from New_shapenet import PartNormalDataset
from pointtransformer import pointtransformerseg
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.context import ParallelMode
import time
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode, _get_enable_parallel_optimizer)
set_seed(1)

num_classes = 16


class MyWithLossCell(nn.Cell):
    """定义损失网络"""

    def __init__(self, backbone, loss_fn):
        """实例化时传入前向网络和损失函数作为参数"""
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        ti1 = time.time()
        """连接前向网络和损失函数"""
        out = self.backbone(data)
        loss = self.loss_fn(out,label)
        ti2 = time.time()
        print("Loss Net time:",ti2-ti1)
        return self.loss_fn(out, label),out


class MyTrainStep(nn.TrainOneStepCell):
    """定义训练流程"""

    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        

    def construct(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss,output = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        loss = F.depend(loss, self.optimizer(grads))
        return loss,output

def to_categorical(y, num_class):
    Eye = ops.Eye()
    new_y = Eye(num_class, num_class, mindspore.float32)[mindspore.Tensor(y)]
    new_y = mindspore.Tensor(new_y)
    return new_y


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def pointnet_seg_train(args_opt):
    """PointNet segmentation train."""
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath("./")
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("PointNet segmentation train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, "pointtransofrmer"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target, device_id = 2,max_call_depth=20000)

    # Data Pipeline.
    logger.info("preparing data...")
    train_dataset = PartNormalDataset(root=args_opt.data_url, split="train",
                                        normal_channel= True)
    train_ds = ds.GeneratorDataset(train_dataset, ["output","cls", "label"],num_parallel_workers=10,shuffle=True)
    train_ds = train_ds.batch(batch_size=args_opt.batch_size)

    test_dataset = PartNormalDataset(root=args_opt.data_url, split="test", normal_channel=True)
    test_ds = ds.GeneratorDataset(test_dataset, ["output","cls", "label"], num_parallel_workers=10, shuffle=False) 
    test_ds = test_ds.batch(batch_size=args_opt.batch_size)

    steps_per_epoch = train_ds.get_dataset_size()
    step_size = steps_per_epoch
    test_steps_per_epoch = test_ds.get_dataset_size()

    logger.info("initializing...")
    # Create model.
    network = pointtransformerseg()

    # load pretrained ckpt
    # Set learning rate scheduler.
    if args_opt.lr_decay_mode == "cosine_decay_lr":
        lr = nn.cosine_decay_lr(min_lr=args_opt.min_lr,
                                max_lr=args_opt.max_lr,
                                total_step=args_opt.epoch_size * step_size,
                                step_per_epoch=step_size,
                                decay_epoch=args_opt.decay_epoch)
    elif args_opt.lr_decay_mode == "piecewise_constant_lr":
        lr = nn.piecewise_constant_lr(args_opt.milestone, args_opt.learning_rates)

    # Define optimizer.
    logger.info("preparing data...")
    network_opt = nn.Adam(network.trainable_params(), lr, args_opt.momentum)

    # Define loss function.
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = "mean")

    #network_loss = ops.SoftmaxCrossEntropyWithLogits()
    # Init the model.
   # network.set_train(True)
    #net_train = Model(network, loss_fn=network_loss, optimizer=network_opt)
    my_network_loss = MyWithLossCell(network,network_loss)
    my_train = MyTrainStep(my_network_loss,network_opt)
    network_with_loss = nn.WithLossCell(network,network_loss)
    train_net = nn.TrainOneStepCell(network_with_loss,network_opt)
    eval_net = nn.WithEvalCell(network,network_loss)
    train_net_output = TrainWithCell(eval_net,network_opt)
    # net_train = nn.TrainOneStepCell(criterion, network_opt)
    train_net.set_train()
    my_train.set_train()
    train_net_output.set_train()
    #eval_net = nn.WithEvalCell(network,network_loss)
    #eval_net = nn.WithEvalCell(network,network_loss)
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(args_opt.epoch_size):
        mean_correct = []
        loss_list = []
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args_opt.epoch_size))
        train_net.set_train(True)
        my_train.set_train(True)
        train_net_output.set_train(True)


        for points in tqdm(train_ds.create_dict_iterator(), total=steps_per_epoch, smoothing=0.9):
     #       points = points.data.numpy()
#            points[:, :, 0:3] = pointnet_utils.random_scale_point_cloud(points[:, :, 0:3])
 #           points[:, :, 0:3] = pointnet_utils.shift_point_cloud(points[:, :, 0:3])
           # points = ms.Tensor(points)
           # points, target = points.float().cuda(), target.long().cuda()
  #          points = points.transpose(2, 1)
            #print(points["output"].shape,points["label"].shape)
            t1 = time.time()
            point_set = points["output"]
            cls = points["cls"]
            target = points["label"]
            t2 = time.time()
            print("read data time",t2-t1)
            cat = ops.Concat(axis = 2)
            onehot = ops.OneHot()
            _, N, C = point_set.shape
            point = cat((mindspore.Tensor(point_set), mindspore.numpy.tile(
            onehot(mindspore.Tensor(cls, mindspore.int32), 16, mindspore.Tensor(1.0, mindspore.float32),
                    mindspore.Tensor(0.0, mindspore.float32)),
            (1, N, 1))))
            #point = cat((mindspore.Tensor(point_set), mindspore.numpy.tile(onehot(mindspore.Tensor(cls,mindspore.int32), 16, mindspore.Tensor(1.0, mindspore.float32), mindspore.Tensor(0.0, mindspore.float32), (1,N, 1))))
           # seg_pred = network(point)
            target = target.view(-1, 1)[:, 0]
            t3 = time.time()

            print("data chuli  time",t3-t2)
            #print(seg_pred)
           # pred_choice = seg_pred.max(1)[1]
           # correct = ops.Equal()(pred_choice,target).sum()
           # correct = pred_choice.eq(target).sum()
            #mean_correct.append(correct / (args.batch_size * 1024))
#            loss = train_net(point,target)
#            loss_num = loss.asnumpy()
#            loss_list.append(loss_num)
#            _,output,eval_label = eval_net(point,target)
#            loss,output,_ = train_net_output(point,target)
            loss,output = my_train(point,target)
           # print(loss)
            t4 = time.time()
            print("train time",t4-t3)
            pred_choice = output.max(1)[1]
            correct = ops.Equal()(pred_choice,target).sum()
            #correct = pred_choice.eq(target).sum()
            mean_correct.append(correct / (args.batch_size * 1024))
        train_instance_acc = np.mean(mean_correct)
        #log_string('Train accuracy is: %.5f' % train_instance_acc)

        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(args_opt.num_part)]
        total_correct_class = [0 for _ in range(args_opt.num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()

        for batch_id, (points, target) in tqdm(enumerate(test_ds), total=len(test_ds), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points , target = points.float().cuda(), target.long().cuda()
            seg_pred= classifier(points)
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(args_opt.num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_checkpoint(state, savepath)  # 保存模型
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet segmentation train.')
    parser.add_argument('--data_url', default="/home/zhangcan/New_tan/Point-Transformers/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/", help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=251, help='Train epoch size.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points.')
    parser.add_argument('--num_classes', type=int, default=40, help='Number of classification.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./pointnet_cls", help='Location of training outputs.')

    parser.add_argument("--learning_rates", type=list, default=None, help="A list of learning rates.")
    parser.add_argument("--lr_decay_mode", type=str, default="cosine_decay_lr", help="Learning rate decay mode.")
    parser.add_argument("--min_lr", type=float, default=0.00001, help="The min learning rate.")
    parser.add_argument("--max_lr", type=float, default=0.001, help="The max learning rate.")
    parser.add_argument("--decay_epoch", type=int, default=250, help="Number of decay epochs.")
    parser.add_argument("--milestone", type=list, default=None, help="A list of milestone.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the moving average.")

    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')
    parser.add_argument('--download', type=bool, default=False, help='Download ModelNet40 train dataset.')
    parser.add_argument('--use_norm', type=bool, default=False, help='use_norm.')
    parser.add_argument('--num_part', type=int, default=50, help='Number of parts.')

    args = parser.parse_known_args()[0]
    pointnet_seg_train(args)
