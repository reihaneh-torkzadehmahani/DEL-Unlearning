# Deletion by Example Localization (DEL)
"Improved Localized Machine Unlearning Through the Lens of Memorization", Under review at ICML

Deletion by Example Localization (DEL) has two components: a localization strategy that identifies critical parameters for a given set of examples, and a simple unlearning algorithm that finetunes only the critical parameters on the data we want to retain. Our experiments on different datasets, forget sets and metrics reveal that DEL outperforms prior work in producing better trade-offs between unlearning performance and accuracy.

# Running Experiments
The current version supports three dataset/model settings: CIFAR-10/ResNet-18, SVHN/ViT and ImageNet-100/ResNet-50. The supported forget set modes are IID and Non-IID. 
 - (1) Pretrain a model on a dataset:  ./scripts/run_pretrain.sh

 - (2) Construct the forget set and train the Oracle model: ./scripts/run_oracle_train.sh

 - (3) Generate the mask for localized unlearning algorithms (DEL and SalLoc): ./scripts/run_generat_mask.sh

 - (4) Run the unlearning algorithm: ./scripts/run_unlearn.sh 

