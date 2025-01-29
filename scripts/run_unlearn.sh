#!/bin/bash

###----------------------.CIFAR10-ResNet18
dataset='cifar10'
model='resnet18'
learning_rate=0.015
epochs=30
forget_mode='non-iid'
forget_classes=(2 5)
forget_data_dir='./data/cifar10_non-iid_forget_indices.pth'
pretrained_dir='./data/pretrained_model_resnet18_cifar10.pth'

###----------------------.SVHN-ViT
#dataset='svhn'
#model='vit'
#learning_rate=0.05
#epochs=30
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
mask_dir='./data/mask_resnet18_cifar10_non-iid_0.1_weighted_grad_channel_thr_0.3.pt'
unlearning_alg='reset+finetune'

for r in {1..3}
do
    python unlearn.py  --dataset $dataset\
	    				         --model $model\
    					         --pretrained_dir $pretrained_dir \
                       --forget_data_dir $forget_data_dir \
	    				         --forget_mode $forget_mode \
	    				         --forget_ratio $forget_ratio\
	    				         --forget_classes ${forget_classes[@]}\
	    				         --learning_rate $learning_rate\
    					         --mask_dir $mask_dir \
    					         --unlearning_alg $unlearning_alg \
	    	               --epochs $epochs \
								       --run $r \



done
