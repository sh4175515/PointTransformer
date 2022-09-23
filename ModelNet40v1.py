import os
import numpy as np
import mindspore.dataset as ds
np.random.seed(1234)

__all__ = ["ModelNet40Dataset"]

class ModelNet40Dataset:
    def __init__(self, root_path, split, use_norm, num_points):
        self.path = root_path
        self.split = split
        self.use_norm = use_norm
        self.num_points = num_points
        shapeid_path = "modelnet40_train.txt" if self.split == "train" else "modelnet40_test.txt"
        catfile = os.path.join(self.path, "modelnet40_shape_names.txt")
        cat = [line.rstrip() for line in open(catfile)]
        self.classes = dict(zip(cat, range(len(cat))))
        shape_ids = [line.rstrip() for line in open(os.path.join(self.path, shapeid_path))]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
        self.datapath = [(shape_names[i], os.path.join(self.path, shape_names[i], shape_ids[i]) + '.txt') for i
                    in range(len(shape_ids))]
        
    def __getitem__(self, index):
        fn = self.datapath[index]
        label = self.classes[self.datapath[index][0]]
        label = np.asarray([label]).astype(np.int32)
        point_cloud = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        if self.use_norm:
            point_cloud = point_cloud[:self.num_points, :]
        else:
            point_cloud = point_cloud[:self.num_points, :3]
        point_cloud[:, :3] = self._pc_normalize(point_cloud[:, :3])
        return point_cloud, label[0]
    
    def __len__(self):
        return len(self.datapath)
    
    def _pc_normalize(self, data):
        centroid = np.mean(data, axis=0)
        data = data - centroid
        m = np.max(np.sqrt(np.sum(data ** 2, axis=1)))
        data /= m
        return data
    
    
if __name__ == "__main__":
    dataset_generator = ModelNet40Dataset(root_path= "/home/cxh/文档/ms3d/ModelNet40/",
                                         split = 'train',
                                         use_norm = False,
                                         num_points = 1024)
    data = dataset_generator[0]
    points=data[0]
    label=data[1]
    #print(data[1].shape)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    dataset = dataset.batch(1)
    for data in dataset.create_dict_iterator():
        pointcloud = data['data'].asnumpy()
        label = data['label'].asnumpy()
        print(pointcloud.shape)
        print(label.shape)
    print('data size:', dataset.get_dataset_size())
        
        