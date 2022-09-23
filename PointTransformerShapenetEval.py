import argparse
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train import Model
from New_shapenet import PartNormalDataset
from pointtransformer import pointtransformercls, pointtransformerseg
import mindspore.dataset as ds
import mindspore



def PointTransformer_Eval(args_opt):
    """PointTransformer eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
    

    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    test_metrics = {}
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(50)]
    total_correct_class = [0 for _ in range(50)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    print(shape_ious)
    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    

    # Data pipeline.
    dataset = PartNormalDataset(root=args_opt.data_url, split="val", normal_channel=True)
    dataset_val = ds.GeneratorDataset(dataset, ["data", "cls", "label"], shuffle=False)
    dataset_val = dataset_val.batch(batch_size=args_opt.batch_size, drop_remainder=True)
    print(dataset_val.get_dataset_size())
    # Create model.
    network = pointtransformerseg()

    # Load checkpoint file for ST test.

    param_dict = load_checkpoint(args_opt.ckpt_file)
    load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    # Begin to eval
    network.set_train(False)
    
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    for _, data in enumerate(dataset_val.create_dict_iterator()):
        points, cls, target = data['data'],data["cls"], data['label']
        
        print(_)
        cat = ops.Concat(axis = 2)
        onehot = ops.OneHot()
        B, N, C = points.shape
        point = cat((mindspore.Tensor(points), mindspore.numpy.tile(
            onehot(mindspore.Tensor(cls, mindspore.int32), 16, mindspore.Tensor(1.0, mindspore.float32),
                    mindspore.Tensor(0.0, mindspore.float32)),
            (1, N, 1))))
        pred = network(point)  # pred.shape=[80000,4]
        
        pred_choice = ops.ArgMaxWithValue(axis=1)(pred)[0]
        pred_np = pred_choice.asnumpy()
        cur_pred_val = pred.view(B,N,50)
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((B, N)).astype(np.int32)
        target_np = target.asnumpy()
        for i in range(B):
            cat = seg_label_to_cat[target_np[i, 0]]
            logits = cur_pred_val_logits[i, :,:]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]],1) + seg_classes[cat][0]

        correct = np.sum(cur_pred_val == target)
        total_correct += correct
        total_seen += (B * N)
        for l in range(50):
            total_seen_class[l] += np.sum(target_np == l)
            total_correct_class[l] += (np.sum((cur_pred_val == l) & (target_np == l)))

        for i in range(B):
            segp = cur_pred_val[i, :]
            segl = target_np[i, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and ( np.sum(segp == l) == 0):
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
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
                print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
    print('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
             1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet eval.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', default= r'/home/zhangcan/New_tan/Point-Transformers/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
                        ,help='Location of data.')
    parser.add_argument('--download', type=bool, default=False, help='Download ModelNet40 val dataset.')
    parser.add_argument('--ckpt_file', type=str,
                        default="/home/zhangcan/seg.ckpt",
                        help='Path of the check point file.')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of batch size.')
    # parser.add_argument('--num_classes', type=int, default=40, help='Number of classification.')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points.')
    parser.add_argument('--use_norm', type=bool, default=True, help='use_norm.')
    parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")

    args = parser.parse_known_args()[0]
    PointTransformer_Eval(args)

