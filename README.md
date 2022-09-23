# Point-Transformer-Mindspore
This repo is a mindspore implementation for Point-Transformer. [Zhao et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)
![Point-Transformer structre](https://github.com/sh4175515/Point-transformer-Mindspore/blob/main/PointTransformer.png)
# Requiremernt
```
python = 3.8
mindspore = 1.7
```
# Classification
Download dataset ModelNet40, save in 
```modelnet40_normal_resampled```

## Train

```
python PointTransformerModelnet40Train.py
```

## Eval

```
python PointTransformerModelnet40Eval.py
```

## Result

The accuracy of Point-Transformer in Modelnet40 is 0.918

# Part Segmentation

Download ShapeNet, save in ```shapenetcore_partanno_segmentation_benchmark_v0_normal```.
## Train
```
python PointTransformerShapenetTrain.py
```
## Eval

```
python PointTransformerShapenetEval.py
```
## Result
The inctance mIOU of Point-Transformer in Shapenet is 0.85

# Acknowledgements
Our implementation is mainly based on the following codebases. 
We gratefully thank the authors for their wonderful works.
[qq456cvb, Point-Transformer](https://github.com/qq456cvb/Point-Transformers); 
[HelloMaroon, Point-Transformer-classification](https://github.com/HelloMaroon/Point-Transformer-classification); 
[pierrefdz,point-transformer](https://github.com/pierrefdz/point-transformer)

