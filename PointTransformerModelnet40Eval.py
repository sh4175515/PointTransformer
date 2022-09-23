
""" PointTransformer eval script. """

import argparse
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train import Model
from dataset.ModelNet40v1 import ModelNet40Dataset
from models.Point_Transformer import PointTransformerCls
import mindspore.dataset as ds


def PointTransformer_Eval(args_opt):
    """PointTransformer eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    # Data pipeline.
    dataset = ModelNet40Dataset(root_path=args_opt.data_url,
                                split='val',
                                use_norm=args_opt.use_norm,
                                num_points=args_opt.num_points)

    dataset_val = ds.GeneratorDataset(dataset, ["data", "label"], shuffle=False)
    dataset_val = dataset_val.batch(batch_size=2, drop_remainder=True)

    # Create model.
    network = PointTransformerCls()

    # Load checkpoint file for ST test.

    param_dict = load_checkpoint(args_opt.ckpt_file)
    load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # Define eval metrics.
    metrics = {"Accuracy": nn.Accuracy()}

    # Init the model.
    model = Model(network, network_loss, metrics=metrics)

    # Begin to eval
    result = model.eval(dataset_val, dataset_sink_mode=True)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet eval.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', help='Location of data.')
    parser.add_argument('--download', type=bool, default=False, help='Download ModelNet40 val dataset.')
    parser.add_argument('--ckpt_file', type=str, help='Path of the check point file.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size.')
    parser.add_argument('--num_classes', type=int, default=40, help='Number of classification.')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points.')
    parser.add_argument('--use_norm', type=bool, default=True, help='use_norm.')
    args = parser.parse_known_args()[0]
    PointTransformer_Eval(args)
