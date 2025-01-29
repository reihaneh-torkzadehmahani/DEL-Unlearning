# DEL-Unlearning
"Improved Localized Machine Unlearning Through the Lens of Memorization", Under review at ICML

# Running Experiments
The current version supports three dataset/model settings: CIFAR-10/ResNet-18, SVHN/ViT and ImageNet-100/ResNet-50. The supported forget set modes are IID and Non-IID. 
 - (1) Pretrain a model on a dataset:  ./scripts/run_pretrain.sh

 - (2) Construct the forget set and train the Oracle model: ./scripts/run_oracle_train.sh

 - (3) Generate the mask for localized unlearning algorithms (DEL and SalLoc): ./scripts/run_generat_mask.sh

 - (4) Run the unlearning algorithm: ./scripts/run_unlearn.sh 

