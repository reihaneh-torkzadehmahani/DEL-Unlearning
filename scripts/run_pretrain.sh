#!/bin/bash

###----------------------.CIFAR10-ResNet18
dataset='cifar10'
model='resnet18'
learning_rate=0.1
epochs=30

###----------------------.SVHN-ViT
#dataset='svhn'
#model='vit'
#learning_rate=0.05
#epochs=30

###----------------------.ImageNet100-ResNet50
#dataset='imagenet100'
#model='resnet50'
#learning_rate=0.1
#epochs=100

batch_size=128
weight_decay=0
base_dir='./data/'

python pretrain.py  --dataset $dataset \
		    --model $model \
		    --batch_size $batch_size \
		    --learning_rate $learning_rate \
		    --weight_decay $weight_decay \
		    --epochs $epochs \
		    --base_dir $base_dir \
