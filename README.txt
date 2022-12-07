#####################
Dataset Preparation
#####################
Please go to https://github.com/taesungp/contrastive-unpaired-translation to prepare the CITYSCAPES dataset following their instruction.



#####################
Training
#####################

For our model, run
=======================
python train.py --dataroot=../../DA/data/cityscapes --direction=BtoA --lambda_var=0.01
=======================

For the base-gan model, run
=======================
python train.py --dataroot=../../DA/data/cityscapes --direction=BtoA --model=base_gan
=======================



#####################
Our modification
#####################
Our code is built based on the implmentation of CUT. Our main contribution are in following lines.

--- models/vista_gan_model.py Line 179-213
--- models/networks.py Line 600-658

