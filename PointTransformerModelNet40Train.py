""" PointNet classification training script."""

import argparse
import mindspore.dataset as ds
import mindspore.nn as nn

from mindspore import context
from mindspore.common import set_seed
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from dataset.ModelNet40v1 import ModelNet40Dataset
from models.Point_Transformer import pointtransformercls
from ms3d.engine.callback import ValAccMonitor

set_seed(1)


def Point_Transformer_train(args_opt):
    """Point_Transformer_train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    dataset = ModelNet40Dataset(root_path=args_opt.data_url,
                                split="train",
                                num_points=args_opt.num_points,
                                use_norm=args_opt.use_norm)

    dataset_train = ds.GeneratorDataset(dataset, ["data", "label"], shuffle=True)
    dataset_train = dataset_train.batch(batch_size=args_opt.batch_size, drop_remainder=True)

    dataset = ModelNet40Dataset(root_path=args_opt.data_url,
                                split="val",
                                num_points=args_opt.num_points,
                                use_norm=args_opt.use_norm)

    dataset_val = ds.GeneratorDataset(dataset, ["data", "label"], shuffle=True)
    dataset_val = dataset_val.batch(batch_size=args_opt.batch_size, drop_remainder=True)

    step_size = dataset_train.get_dataset_size()

    # Create model.
    network = pointtransformercls()

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
    network_opt = nn.Adam(network.trainable_params(), lr, args_opt.momentum)

    # Define loss function.
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # Define metrics.
    metrics = {"Accuracy": nn.Accuracy()}

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size,
                                   keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='Point_Transformer_Cls',
                                    directory=args_opt.ckpt_save_dir,
                                    config=ckpt_config)

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, ValAccMonitor(model, dataset_val, args_opt.epoch_size)],
                dataset_sink_mode=args_opt.dataset_sink_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point_Transformer train.')
    parser.add_argument('--data_url', default="./modelnet40_normal_resampled", help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=250, help='Train epoch size.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points.')
    parser.add_argument('--num_classes', type=int, default=40, help='Number of classification.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./Point_Transformer_cls",
                        help='Location of training outputs.')
    parser.add_argument("--learning_rates", type=list, default=None, help="A list of learning rates.")
    parser.add_argument("--lr_decay_mode", type=str, default="cosine_decay_lr", help="Learning rate decay mode.")
    parser.add_argument("--min_lr", type=float, default=0.00001, help="The min learning rate.")
    parser.add_argument("--max_lr", type=float, default=0.1, help="The max learning rate.")
    parser.add_argument("--decay_epoch", type=int, default=250, help="Number of decay epochs.")
    parser.add_argument("--milestone", type=list, default=None, help="A list of milestone.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the moving average.")
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')
    parser.add_argument('--download', type=bool, default=False, help='Download ModelNet40 train dataset.')
    parser.add_argument('--use_norm', type=bool, default=True, help='use_norm.')

    args = parser.parse_known_args()[0]
    Point_Transformer_train(args)
