#!/bin/bash

###----------------------.CIFAR10-ResNet18
dataset='cifar10'
model='resnet18'
learning_rate=0.1
epochs=30
forget_mode='non-iid'
forget_classes=(2 5)

###----------------------.SVHN-ViT
#dataset='svhn'
#model='vit'
#learning_rate=0.05
#epochs=3
#forget_mode='non-iid'
#forget_classes=(3 6)
#forget_data_dir='./data/svhn_forget_indices.pth'


###----------------------.ImageNet100-ResNet50
#dataset='imagenet100'
#model='resnet50'
#learning_rate=0.1
#epochs=100
#forget_data_dir='./data/imagenet100_forget_indices.pth'

###----------------------- General parameters
batch_size=128
weight_decay=0
forget_ratio=0.1
#forget_mode='iid'
#forget_classes=(-1)
base_dir='./data/'

for r in {1..3}
do
  python oracle_train.py  --dataset $dataset \
		                      --forget_ratio $forget_ratio\
		                      --forget_mode $forget_mode\
		                      --forget_classes "$(IFS=,; echo "${forget_classes[*]}")"\
                          --model $model \
                          --batch_size $batch_size \
                          --learning_rate $learning_rate \
                          --weight_decay $weight_decay \
		                      --epochs $epochs \
                          --base_dir $base_dir \
                          --run $r \

done
