#!/bin/bash

###----------------------.CIFAR10-ResNet18
dataset='cifar10'
model='resnet18'
#forget_mode='non-iid'
#forget_classes=(2 5)
forget_mode='iid'
forget_classes=(-1)
forget_data_dir='./data/cifar10_forget_indices.pth'
pretrained_dir='./data/pretrained_model_resnet18_cifar10.pth'

###----------------------.SVHN-ViT
#dataset='svhn'
#model='vit'
#forget_mode='non-iid'
#forget_classes=(3 6)
#forget_data_dir='./data/svhn_forget_indices.pth'
#pretrained_dir='./data/pretrained_model_vit_svhn.pth'


###----------------------.ImageNet100-ResNet50
#dataset='imagenet100'
#model='resnet50'
#forget_data_dir='./data/imagenet100_forget_indices.pth'
#pretrained_dir='./data/pretrained_model_resnet50_imagenet100.pth'

###----------------------- General parameters
#---- Salun mask
critic_criteria='grad'
granularity='param'
threshold=(1 3)
forget_ratio=0.1

python generate_mask.py --model $model\
                        --dataset $dataset\
                        --forget_data_dir $forget_data_dir\
		                    --forget_ratio $forget_ratio\
		                    --forget_mode $forget_mode\
		                    --forget_classes ${forget_classes[@]}\
                        --pretrained_dir $pretrained_dir\
                        --critic_criteria $critic_criteria\
                        --granularity $granularity\
                        --threshold ${threshold[@]} \
