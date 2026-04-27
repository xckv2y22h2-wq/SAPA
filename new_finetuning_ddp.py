from __future__ import print_function

import argparse
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tqdm import tqdm
import time
import datetime
import random
import warnings
# import wandb
import copy

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# from torchvision.datasets import StanfordCars, Food101, SUN397, EuroSAT, \
#     Caltech256, Country211, Flowers102, PCAM, FGVCAircraft

from replace.tv_datasets import StanfordCars, Food101, SUN397, EuroSAT, DTD, \
    Caltech101, Caltech256, Country211, Flowers102, PCAM, FGVCAircraft, OxfordIIITPet

from torchvision.datasets import CIFAR10, CIFAR100, STL10

import torchvision.transforms as transforms
import torchvision

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from modified_clip import clip
from models.model import *
from models.prompters import TokenPrompter, NullPrompter, PromptLearner, NonePrompter
from models.swa import *
from attacks import *
from adv_clip_loss import *


from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint, str2bool
from utils import cosine_lr, convert_models_to_fp32, refine_classname
from utils import reset_log_file, log_record

from data_utils.autoaugment import ImageNetPolicy

import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import functools
from autoattack import AutoAttack

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
Default Training Setting: 
Batch_size=256, Dataset=ImageNet, train_numsteps=10, train_stepsize==1, train_eps=2
learning_rate=1e-5  Optimizer=adamw

Default Evaluation Setting: 
20-step PGD  & CW & AA

# Before Training, run the following codes to download CLIP-ViT-B/L
python download_CLIP.py



## 2025.02.19
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=59660 new_finetuning_ddp.py --arch ViT-L/14 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TeCoA --exp_name FT_TeCoA_VITL_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=59661 new_finetuning_ddp.py --arch ViT-L/14 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method FARE --exp_name FT_FARE_VITL_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=59662 new_finetuning_ddp.py --arch ViT-L/14 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --weight_decay 1e-4 --method FARE --exp_name FT_FARE_VITL_epoch10_eps2_10steps_wd1e4_adamw

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=1 --master_port=59663 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TeCoA --exp_name FT_TeCoA_VITB_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=1 --master_port=59664 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method FARE --exp_name FT_FARE_VITB_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=1 --master_port=59665 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --weight_decay 1e-4 --method FARE --exp_name FT_FARE_VITB_epoch10_eps2_10steps_wd1e4_adamw

## 2025.02.25
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=58510 new_finetuning_ddp.py --arch ViT-L/14 --epochs 10 --train_eps 2 --train_numsteps 10 --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --optim adamw --batch_size 128 --learning_rate 5e-6 --exp_name FT_PMG_VITL_epoch10_eps2_10steps_adamw_bs128_lr5e6
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=58511 new_finetuning_ddp.py --arch ViT-L/14 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_VITL_epoch10_eps2_10steps_adamw

######################################### ViT-Base #########################################
# DDP Adv FT (TeCoA)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29518 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --method TeCoA --exp_name FT_TeCoA_VITB_epoch10_eps2_10steps
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29510 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TeCoA --exp_name FT_TeCoA_VITB_epoch10_eps2_10steps_adamw

# DDP Adv FT (PMG)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29519 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_VITB_epoch10_eps2_10steps
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=28849 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_VITB_epoch10_eps2_10steps_adamw

# DDP Adv FT (FARE)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29520 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --weight_decay 1e-4 --method FARE --exp_name FT_FARE_VITB_epoch10_eps2_10steps

# DDP Adv FT (TRADES)
CUDA_VISIBLE_DEVICES=2,4 torchrun --nproc_per_node=2 --master_port=29521 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_VITB_epoch10_eps2_10steps
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29019 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_VITB_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29019 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --weight_decay 1e-4 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_VITB_epoch10_eps2_10steps_adamw_wd1e4
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=27825 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TRADES --W_Pred_Align_Ori 6.0 --exp_name FT_TRADES60_VITB_epoch10_eps2_10steps_adamw
To run:
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=27826 new_finetuning_ddp.py --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TRADES --W_Pred_Align_Ori 3.0 --exp_name FT_TRADES30_VITB_epoch10_eps2_10steps_adamw
######################################### ViT-Base #########################################




######################################### ViT-Large #########################################

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29666 new_finetuning_ddp.py --arch ViT-L/14 --epochs 2 --train_eps 2 --train_numsteps 10 --method TeCoA --exp_name FT_TeCoA_VITL_epoch2_eps2_10steps
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 --master_port=29834 new_finetuning_ddp.py --arch ViT-L/14 --epochs 2 --train_eps 2 --train_numsteps 10 --optim adamw --method TeCoA --exp_name FT_TeCoA_VITL_epoch2_eps2_10steps_adamw
To run:
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29699 new_finetuning_ddp.py --arch ViT-L/14 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TeCoA --exp_name FT_TeCoA_VITL_epoch10_eps2_10steps_adamw

######################################### ViT-Large #########################################




######################################### ResNet-50/101 #########################################

### ResNet-50
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29508 new_finetuning_ddp.py --arch RN50 --epochs 10 --train_eps 2 --train_numsteps 10 --method TeCoA --exp_name FT_TeCoA_RN50_epoch10_eps2_10steps
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 new_finetuning_ddp.py --arch RN50 --epochs 10 --train_eps 2 --train_numsteps 10 --method TeCoA --optim adamw --exp_name FT_TeCoA_RN50_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29501 new_finetuning_ddp.py --arch RN50 --epochs 10 --train_eps 2 --train_numsteps 10 --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_RN50_epoch10_eps2_10steps
CUDA_VISIBLE_DEVICES=1,7 torchrun --nproc_per_node=2 --master_port=29517 new_finetuning_ddp.py --arch RN50 --epochs 10 --train_eps 2 --train_numsteps 10 --method PMG --optim adamw --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_RN50_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29508 new_finetuning_ddp.py --arch RN50 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --weight_decay 1e-4 --method FARE --exp_name FT_FARE_RN50_epoch10_eps2_10steps
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29508 new_finetuning_ddp.py --arch RN50 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_RN50_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29508 new_finetuning_ddp.py --arch RN50 --epochs 10 --train_eps 2 --train_numsteps 10 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_RN50_epoch10_eps2_10steps


### ResNet-101
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29506 new_finetuning_ddp.py --arch RN101 --epochs 10 --train_eps 2 --train_numsteps 10 --method TeCoA --exp_name FT_TeCoA_RN101_epoch10_eps2_10steps
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29506 new_finetuning_ddp.py --arch RN101 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TeCoA --exp_name FT_TeCoA_RN101_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29867 new_finetuning_ddp.py --arch RN101 --epochs 10 --train_eps 2 --train_numsteps 10 --method PMG --optim adamw --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_RN101_epoch10_eps2_10steps_adamw
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29506 new_finetuning_ddp.py --arch RN101 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --weight_decay 1e-4 --method FARE --exp_name FT_FARE_RN101_epoch10_eps2_10steps
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29508 new_finetuning_ddp.py --arch RN101 --epochs 10 --train_eps 2 --train_numsteps 10 --optim adamw --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_RN101_epoch10_eps2_10steps_adamw




######################################### ResNet-50/101 #########################################




# Robustness evaluation (No need for DDP)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29521 new_finetuning_ddp.py --evaluate --eval_type full --test_eps 1 --resume XXXX/model_best.pth.tar
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29521 new_finetuning_ddp.py --evaluate --eval_type full --arch ViT-L/14 --test_eps 1 --batch_size 128 --resume XXXX/model_best.pth.tar

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=62121 new_finetuning_ddp.py --evaluate --eval_type temp --test_eps 1 --resume Source/TeCoAmodel_best.pth.tar





---------------------------------------------------------------------------------------------------

# TeCoA (ICLR 2023)
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 1 --method TeCoA --exp_name FT_TeCoA_epoch10_eps1

# PMG (CVPR 2024)
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 1 --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_epoch10_eps1

# FARE (ICML 2024)
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 1 --method FARE --exp_name FT_FARE_epoch10_eps1
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 2 --train_eps 1 --train_numsteps 10 --method FARE --exp_name FT_FARE_epoch2_eps1_10steps_sgd_wd0
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 2 --train_eps 1 --train_numsteps 10 --optim adamw --method FARE --exp_name FT_FARE_epoch2_eps1_10steps_adamw_wd0
CUDA_VISIBLE_DEVICES=2,3 python new_finetuning.py --epochs 2 --train_eps 1 --train_numsteps 10 --optim adamw --weight_decay 1e-4 --method FARE --exp_name FT_FARE_epoch2_eps1_10steps_adamw_wd1e4


# Adv_Img + Adv_Text (Joint tuning PGD-AT)
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 1 --method ImgText_PGD --W_Pred_Align_Ori 1.0 --adv_prompt_gen True --text_perb_stepsize 1e-4 --exp_name FT_ImgText_PGD_epoch10_eps1

# TRADES (Nat_CE + beta * KL(adv, ori_nat))
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_epoch10_eps1
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --mul_noise_beta 0.05 --exp_name FT_TRADES90_mulbeta005_epoch10_eps1

# 2024.05.03 TRADES Weight decay
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --weight_decay 1e-4 --epochs 20 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_wd1e4_epoch20_eps1
CUDA_VISIBLE_DEVICES=2,3 python new_finetuning.py --weight_decay 1e-3 --epochs 20 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_wd1e3_epoch20_eps1
CUDA_VISIBLE_DEVICES=4,5 python new_finetuning.py --weight_decay 1e-2 --epochs 20 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_wd1e2_epoch20_eps1
CUDA_VISIBLE_DEVICES=6,7 python new_finetuning.py --weight_decay 1e-1 --epochs 20 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_wd1e1_epoch20_eps1

# 2024.05.04
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 20 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --swa 50 --exp_name FT_TRADES90_swa50_epoch20_eps1
CUDA_VISIBLE_DEVICES=2,3 python new_finetuning.py --epochs 20 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --swa 60 --exp_name FT_TRADES90_swa60_epoch20_eps1
CUDA_VISIBLE_DEVICES=4,5 python new_finetuning.py --epochs 20 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --swa 70 --exp_name FT_TRADES90_swa70_epoch20_eps1
CUDA_VISIBLE_DEVICES=6,7 python new_finetuning.py --epochs 20 --train_eps 1 --method TRADES --W_Pred_Align_Ori 9.0 --swa 80 --exp_name FT_TRADES90_swa80_epoch20_eps1

# 2024.05.06 TRADES with more iterations + swa
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 20 --train_eps 1 --train_numsteps 5 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_epoch20_5steps_eps1
CUDA_VISIBLE_DEVICES=2,3 python new_finetuning.py --epochs 20 --train_eps 1 --train_numsteps 5 --method TRADES --W_Pred_Align_Ori 9.0 --swa 50 --exp_name FT_TRADES90_swa50_epoch20_5steps_eps1
CUDA_VISIBLE_DEVICES=4,5 python new_finetuning.py --epochs 20 --train_eps 1 --train_numsteps 10 --method TRADES --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_epoch20_10steps_eps1
CUDA_VISIBLE_DEVICES=6,7 python new_finetuning.py --epochs 20 --train_eps 1 --train_numsteps 10 --method TRADES --W_Pred_Align_Ori 9.0 --swa 50 --exp_name FT_TRADES90_swa50_epoch20_10steps_eps1

-----------------------------------------------------------
New Project:

# 2024.10.20/24/28/11.1/11.4
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 1 --train_numsteps 5 --method TRADES --optim adamw --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_epoch10_5steps_eps1_adamw_wd0
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 1 --train_numsteps 5 --method TRADES --optim adamw --weight_decay 1e-4 --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_epoch10_5steps_eps1_adamw_wd1e4
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 2 --train_numsteps 5 --method TRADES --optim adamw --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_epoch10_5steps_eps2_adamw_wd0
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 3 --train_numsteps 5 --method TRADES --optim adamw --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_epoch10_5steps_eps3_adamw_wd0
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 4 --train_numsteps 5 --method TRADES --optim adamw --W_Pred_Align_Ori 9.0 --exp_name FT_TRADES90_epoch10_5steps_eps4_adamw_wd0

---------------------------------
More Epsilon AT:
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 3 --train_stepsize 3 --method TeCoA --exp_name FT_TeCoA_Eps3
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 3 --train_stepsize 3 --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_Eps3
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --epochs 10 --train_eps 3 --train_stepsize 3 --method FARE --exp_name FT_FARE_Eps3

---------------------------------
Other Architectures:
RN-50:
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --arch RN50 --epochs 10 --train_eps 1 --method TeCoA --exp_name FT_TeCoA_RN50_epoch10_eps1
CUDA_VISIBLE_DEVICES=2,3 python new_finetuning.py --arch RN50 --epochs 10 --train_eps 1 --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_RN50_epoch10_eps1

ViT-L:  # 2024-10-20/25/11.1
CUDA_VISIBLE_DEVICES=0,1,2,3 python new_finetuning.py --arch ViT-L/14 --epochs 10 --train_eps 1 --train_numsteps 2 --method TeCoA --exp_name FT_TeCoA_VITL_epoch10_eps1_2steps
CUDA_VISIBLE_DEVICES=0,1,2,3 python new_finetuning.py --arch ViT-L/14 --epochs 2 --train_eps 1 --train_numsteps 10 --method TeCoA --exp_name FT_TeCoA_VITL_epoch2_eps1_10steps
CUDA_VISIBLE_DEVICES=0,1,2,3 python new_finetuning.py --arch ViT-L/14 --epochs 10 --batch_size 128 --train_eps 1 --train_numsteps 2 --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_VITL_epoch10_bs128_eps1_2steps
To run
CUDA_VISIBLE_DEVICES=0,1,2,3 python new_finetuning.py --arch ViT-L/14 --epochs 2 --train_eps 1 --train_numsteps 10 --method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name FT_PMG_VITL_epoch2_eps1_10steps

---------------------------------
Evaluation:  
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --evaluate --eval_type full --test_eps 1 --resume XXXX/model_best.pth.tar
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --evaluate --eval_type full --test_eps 1 --resume Source_PT/TeCoAmodel_best.pth.tar
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --evaluate --eval_type full --attack CW --test_eps 1 --resume XXXX/model_best.pth.tar

Evaluation -- different templates
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --evaluate --eval_type full --attack pgd --eval_template 1 --test_eps 1 --resume Source_PT/TeCoAmodel_best.pth.tar

Evaluation -- different architectures
CUDA_VISIBLE_DEVICES=0,1 python new_finetuning.py --evaluate --eval_type full --arch RN50 --attack pgd --test_eps 1 --resume XXXX/model_best.pth.tar
CUDA_VISIBLE_DEVICES=0,1,2,3 python new_finetuning.py --evaluate --eval_type full --arch ViT-L/14 --test_eps 1 --resume XXXX/model_best.pth.tar

"""

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=2000,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=1,
                        help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epoch5s')
    parser.add_argument("--mix_alpha", type=float, default=-1,
                        help="interpolation")

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use: sgd|adamw')
    parser.add_argument('--learning_rate', type=float, default=1e-5,  ## Change from 1e-7 to 1e-5
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--train_eps', type=float, default=2,
                        help='momentum')
    parser.add_argument('--train_numsteps', type=int, default=2)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_eps', type=float, default=2,
                        help='momentum')
    parser.add_argument('--test_numsteps', type=int, default=20)
    parser.add_argument('--test_stepsize', type=int, default=1)
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--imagenet_root', type=str, default='temp')
    parser.add_argument('--arch', type=str, default='ViT-B/32',
                        help='ViT-B/32|RN50|RN101|ViT-L/14')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=0,
                        help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./datasets/',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='ImageNet',
                        help='Pre-training Dataset: cifar10|cifar100|ImageNet')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='../save_ckpts',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    
    parser.add_argument('--CW', action='store_true')
    parser.add_argument('--autoattack', action='store_true')
    parser.add_argument('--attack', choices=['pgd', 'CW', 'AA'], default='pgd')

    
    parser.add_argument('--train_class_count', type=int, default=90)
    parser.add_argument('--last_num_ft', type=int, default=-1)
    parser.add_argument('--noimginprop', action='store_true')
    parser.add_argument('--exp_name', type=str, default=None)

    # Augmentation + SWA
    parser.add_argument('--aug_type', type=str, default='Vanilla',
                        help='Vanilla|Vanilla_Flip|Resizecrop|Resizecrop_Flip|Autoaug')
    parser.add_argument('--swa', type=int, default=-1,
                    help='start swa from XX%')

    # Evaluation
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--eval_type', type=str, default="full",
                        help='fast|full')
    parser.add_argument('--eval_template', type=str, default="0",
                        help='0: This is a photo of a X; 1: a photo of a X; 2: a picture of a X; 3: an image of a X; 4: an example of a X')
    
    # Text Prompt Tuning
    parser.add_argument('--adv_prompt_gen', type=str2bool, default="False",
                        help='Whether to conduct adversarial prompt generation')
    parser.add_argument('--ctx', type=int, default=16,
                        help='number of context vector')
    parser.add_argument('--ctx_init', type=str, default='This is a photo of a',
                        help='Initialization for context prompt (e.g., (This is a photo of a)|(a photo of a))')
    parser.add_argument('--position', type=str, default='end',
                        help='CLS prompt position: end|middle|front')
    parser.add_argument('--text_perb_stepsize', type=float, default=0.0, 
                        help='perturbation step size for texts, the perturbation share the same step for adv images default 3e-4')
    
    # Extra modules
    parser.add_argument('--method', type=str, default='TRADES',
                        help='TeCoA|PMG|FARE|ImgText_PGD|TRADES')
    parser.add_argument('--W_Pred_Align', type=float, default=0.0,
                        help='Prediction alignment between clean and adv logits')
    parser.add_argument('--W_Pred_Align_Ori', type=float, default=0.0,
                        help='Prediction alignment between adv logits to the original clip-clean logits')

    ## Noise Modulation para
    parser.add_argument('--mul_noise_beta', type=float, default=0.0,
                        help='multiplicative_noise -- std')
    

    args = parser.parse_args()

    if args.evaluate:
        args.filename = args.resume.split('/')[0]
    else:
        args.filename = args.exp_name

    return args


best_acc1 = 0

###### DDP Init ######
dist.init_process_group(
    backend='nccl', 
    init_method='env://',
    timeout=datetime.timedelta(seconds=1200)
)
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
###### DDP Init ######




def train(train_loader, texts, model, original_model, prompter, add_prompter,
          optimizer, scheduler, criterion, scaler, epoch, prompt_learner, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.module.visual.train()

    num_batches_per_epoch = len(train_loader)

    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    # print('text token', texts)

    end = time.time()

    # original prompter state
    if args.adv_prompt_gen:
        original_prompter_state = copy.deepcopy(prompt_learner.state_dict())
        args.original_prompter_state = original_prompter_state

    for i, (images, target) in enumerate(tqdm(train_loader, ncols = 80)):

        # measure data loading time
        data_time.update(time.time() - end)

        BATCH_SIZE = images.size(0)
        # print('bs', BATCH_SIZE)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        # print(images.min(), images.max())

        # with automatic mixed precision
        with autocast():
            
            if args.method == 'TeCoA':
                loss, pred_adv = FT_TeCoA_loss(images, target, text_tokens, optimizer, model, original_model,
                                               prompter, add_prompter, prompt_learner, args)
            elif args.method == 'PMG':
                loss, pred_adv = FT_PMG_loss(images, target, text_tokens, optimizer, model, original_model,
                                             prompter, add_prompter, prompt_learner, args)
            elif args.method == 'ImgText_PGD':
                loss, pred_adv = FT_ImgText_PGD_loss(images, target, text_tokens, optimizer, model, original_model,
                                                     prompter, add_prompter, prompt_learner, args)
            elif args.method == 'TRADES':
                loss, pred_adv = FT_TRADES_loss(images, target, text_tokens, optimizer, model, original_model,
                                                prompter, add_prompter, prompt_learner, args)
            elif args.method == 'FARE':
                loss, pred_adv = FT_FARE_loss(images, target, text_tokens, optimizer, model, original_model,
                                                prompter, add_prompter, prompt_learner, args)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(pred_adv, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and i != 0:
            if dist.get_rank() == 0:
                progress.display(i)

        if i % args.save_freq == 0:
            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': prompter.state_dict(),
                    'add_prompter': add_prompter.state_dict(),
                    'vision_encoder_state_dict': model.module.visual.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, args)

    return losses.avg, top1.avg



def main():
    global best_acc1, device

    args = parse_option()

    args.original_test_eps = args.test_eps
    args.train_eps = args.train_eps / 255.
    args.test_eps = args.test_eps / 255.
    args.train_stepsize = args.train_stepsize / 255.
    args.test_stepsize = args.test_stepsize / 255.

    if args.resume is not None:
        args.resume = os.path.join("../save_ckpts", args.resume)

    if dist.get_rank() == 0:
        print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True

    if not args.root:
        args.root = './datasets/'
    imagenet_root = os.path.join(args.root, "ImageNet")

    imgnet_full = imagenet_root

    # create model
    # add_prompt_len = args.add_prompt_size

    # No prompts during the inference statge
    add_prompt_len = 0

    model, preprocess = clip.load(args.arch, device, jit=False, prompt_len=add_prompt_len)
    # model_text, model_image = None, None

    convert_models_to_fp32(model)
    # model = torch.nn.DataParallel(model)  # .to(device)
    if args.arch == 'RN50' or args.arch == 'RN101':
        if dist.get_world_size() > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
    else:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.eval()
    
    original_model = None
    if args.W_Pred_Align_Ori > 0.0 or args.method == 'FARE':
        original_model, preprocess = clip.load(args.arch, device, jit=False, prompt_len=add_prompt_len)
        convert_models_to_fp32(original_model)
        # original_model = torch.nn.DataParallel(original_model)  # .to(device)
        if args.arch == 'RN50' or args.arch == 'RN101':
            if dist.get_world_size() > 1:
                original_model = nn.SyncBatchNorm.convert_sync_batchnorm(original_model)
            original_model = DDP(original_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        else:
            original_model = DDP(original_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        original_model.eval()


    ###### SWA ######
    if args.swa >= 0:
        swa_model, preprocess = clip.load(args.arch, device, jit=False, prompt_len=add_prompt_len)
        convert_models_to_fp32(swa_model)
        # swa_model = torch.nn.DataParallel(swa_model)  # .to(device)
        if args.arch == 'RN50' or args.arch == 'RN101':
            if dist.get_world_size() > 1:
                swa_model = nn.SyncBatchNorm.convert_sync_batchnorm(swa_model)
            swa_model = DDP(swa_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        else:
            swa_model = DDP(swa_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        swa_model.eval()

        args.swa_start = int((args.swa/100) * args.epochs)
        args.swa_decay = 0.001  
        if dist.get_rank() == 0:    
            print("-" * 10)
            print("Using SWA, starting from {}-th epoch with decay rate {}".format(args.swa_start, args.swa_decay))
            print("-" * 10)
    else:
        swa_model = None
        args.swa_start = 99999
    ###### SWA ######

    ### !!! These two are prompters for the images
    prompter = NullPrompter()  # .to(device)
    add_prompter = TokenPrompter(add_prompt_len)  # .to(device)

    # prompter = torch.nn.DataParallel(prompter).cuda()
    # add_prompter = torch.nn.DataParallel(add_prompter).cuda()
    prompter = prompter.to(device)
    add_prompter = add_prompter.to(device)
    # prompter = DDP(prompter, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # add_prompter = DDP(add_prompter, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # define criterion and optimizer
    # we finetune the image module parameters only
    if args.last_num_ft == -1:
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model.module.visual.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.optim == 'adamw':
            optimizer = torch.optim.AdamW(model.module.visual.parameters(), 
                                          lr=args.learning_rate, 
                                          weight_decay=args.weight_decay)
    else:
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(list(model.module.visual.parameters())[-args.last_num_ft:],
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.optim == 'adamw':
            optimizer = torch.optim.AdamW(list(model.module.visual.parameters())[-args.last_num_ft:], 
                                          lr=args.learning_rate, 
                                          weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    args.start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if dist.get_rank() == 0:
                print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
                pass

            if args.mix_alpha > 0:
                alpha = args.mix_alpha
                # model1, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
                # model2, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
                # model1 = torch.nn.DataParallel(model1)
                # model2 = torch.nn.DataParallel(model2)

                checkpoint_ori = torch.load('original_clip.pth.tar')
                theta_ori = checkpoint_ori['vision_encoder_state_dict']
                theta_rob = checkpoint['vision_encoder_state_dict']

                theta = {
                    key: (1 - alpha) * theta_ori[key] + alpha * theta_rob[key]
                    for key in theta_ori.keys()
                }
                model.module.visual.load_state_dict(theta)

            else:

                model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            # prompter.load_state_dict(checkpoint['state_dict'])
            # add_prompter.load_state_dict(checkpoint['add_prompter'])
            if dist.get_rank() == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            if dist.get_rank() == 0:
                print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    if args.eval_template == '0':
        template = 'This is a photo of a {}'
    elif args.eval_template == '1':
        template = 'a photo of a {}'
    elif args.eval_template == '2':
        template = 'a picture of a {}'
    elif args.eval_template == '3':
        template = 'an image of a {}'
    elif args.eval_template == '4':
        template = 'an example of a {}'
    
    if dist.get_rank() == 0:
        print(f'template: {template}')

    # TODO: we can train on cifar10 and test on cifar10, 100 in zero shot way, to see if generalize.
    preprocess = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
    preprocess224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
    preprocess224_interpolate = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
    ############################ Augmentation  ############################
    preprocess224_vanilla_flip = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    preprocess224_resizecrop = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.ToTensor()
    ])
    preprocess224_resizecrop_flip = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    preprocess_autoaug = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        ImageNetPolicy(), 
        transforms.ToTensor()
    ])
    # Vanilla|Vanilla_Flip|Resizecrop|Resizecrop_Flip|Autoaug
    if args.aug_type == 'Vanilla':
        IN_aug_type = preprocess224
    elif args.aug_type == 'Vanilla_Flip':
        IN_aug_type = preprocess224_vanilla_flip
    elif args.aug_type == 'Resizecrop':
        IN_aug_type = preprocess224_resizecrop
    elif args.aug_type == 'Resizecrop_Flip':
        IN_aug_type = preprocess224_resizecrop_flip
    elif args.aug_type == 'Autoaug':
        IN_aug_type = preprocess_autoaug

    ############################ Augmentation  ############################

    if args.dataset == 'cifar100':
        train_dataset = CIFAR100(args.root, transform=preprocess,
                                 download=True, train=True)

        val_dataset = CIFAR100(args.root, transform=preprocess,
                               download=True, train=False)
    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(args.root, transform=preprocess,
                                download=True, train=True)

        val_dataset = CIFAR10(args.root, transform=preprocess,
                              download=True, train=False)

    elif args.dataset == 'ImageNet':
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(imagenet_root, 'train'),
            transform=IN_aug_type
        )

    val_dataset_list = []
    val_dataset_name = ['StanfordCars', 'Food101', 'PCAM', 'cifar100', 'oxfordpet', 'flowers102',
                        'Country211', 'dtd', 'EuroSAT', 'fgvc_aircraft', 'ImageNet', 'cifar10', 'SUN397']

    if args.evaluate:
        if args.eval_type == 'fast':
            val_dataset_name = ['ImageNet', 'SUN397', 'Food101', 'flowers102', 'Caltech101', 'Caltech256']
        elif args.eval_type == 'full':
            val_dataset_name = ['ImageNet', 'cifar10', 'STL10', 'cifar100', 
                                'SUN397', 'StanfordCars', 'Food101', 'oxfordpet', 
                                'flowers102', 'dtd', 'EuroSAT', 'fgvc_aircraft', 'PCAM', 'Caltech101', 'Caltech256']
        elif args.eval_type == 'motivation':
            val_dataset_name = ['ImageNet', 'Caltech101']
        elif args.eval_type == 'temp':
            val_dataset_name = ['cifar10']
        elif args.eval_type == 'StanfordCars':
            val_dataset_name = ['StanfordCars']
    else:
        val_dataset_name = ['cifar10', 'cifar100', 'dtd', 'EuroSAT']


    for each in val_dataset_name:
        if each == 'cifar10':
            val_dataset_list.append(CIFAR10(args.root, transform=preprocess,
                                            download=True, train=False))
        elif each == 'cifar100':
            val_dataset_list.append(CIFAR100(args.root, transform=preprocess,
                                             download=True, train=False))
        elif each == 'Caltech101':
            val_dataset_list.append(Caltech101(args.root, target_type='category', transform=preprocess224,
                                               download=True))
        elif each == 'PCAM':
            val_dataset_list.append(PCAM(args.root, split='test', transform=preprocess224,
                                         download=True))
        elif each == 'STL10':
            val_dataset_list.append(STL10(args.root, split='test',
                                          transform=preprocess, download=True))
        elif each == 'SUN397':
            val_dataset_list.append(SUN397(args.root,
                                           transform=preprocess224, download=True))
        elif each == 'StanfordCars':
            val_dataset_list.append(StanfordCars(args.root, split='test',
                                                 transform=preprocess224, download=True))
        elif each == 'Food101':
            val_dataset_list.append(Food101(args.root, split='test',
                                            transform=preprocess224, download=True))
        elif each == 'oxfordpet':
            val_dataset_list.append(OxfordIIITPet(args.root, split='test',
                                                  transform=preprocess224, download=True))
        elif each == 'EuroSAT':
            val_dataset_list.append(EuroSAT(args.root,
                                            transform=preprocess224, download=True))

        elif each == 'Caltech256':
            val_dataset_list.append(Caltech256(args.root, transform=preprocess224,
                                               download=True))
        # elif each == 'FER2013':
        #     val_dataset_list.append(OxfordIIITPet(args.root, split='test',
        #                                           transform=preprocess224, download=True))
        elif each == 'flowers102':
            val_dataset_list.append(Flowers102(args.root, split='test',
                                               transform=preprocess224, download=True))
        elif each == 'Country211':
            val_dataset_list.append(Country211(args.root, split='test',
                                               transform=preprocess224, download=True))
        elif each == 'dtd':
            val_dataset_list.append(DTD(args.root, split='test',
                                        transform=preprocess224, download=True))

        elif each == 'fgvc_aircraft':
            val_dataset_list.append(FGVCAircraft(args.root, split='test',
                                                 transform=preprocess224, download=True))
        elif each == 'ImageNet':
            val_dataset_list.append(torchvision.datasets.ImageFolder(
                os.path.join(imgnet_full, 'val'),
                transform=preprocess224))

            # val_dataset_list.append(torchvision.datasets.ImageNet(
            # root=imagenet_root,
            # split='val',
            # transform=preprocess224))

    ############################ Subset to simulate the last batch (For test only) ############################
    # from torch.utils.data import Subset
    # class_names = train_dataset.classes
    # subset_indices = torch.randperm(len(train_dataset))[:143]
    # temp_train_dataset = Subset(train_dataset, subset_indices)
    # temp_train_dataset.classes = train_dataset.classes
    # train_dataset = temp_train_dataset
    ############################ Subset to simulate the last batch ############################

    train_sampler = None
    val_sampler = None

    train_sampler = DistributedSampler(train_dataset)
    # val_sampler = DistributedSampler(val_dataset, shuffle=False) 

    global_batch_size = args.batch_size
    world_size = dist.get_world_size()
    local_batch_size = global_batch_size // world_size

    train_loader = DataLoader(train_dataset,
                              batch_size=local_batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=False, sampler=train_sampler)       # shuffle no need to be True --- Sampler

    val_loader_list = [DataLoader(each,
                                  batch_size=global_batch_size, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False, sampler=val_sampler, drop_last=True) for each in
                       val_dataset_list]

    ### serial number (not semantic classes)
    class_names = train_dataset.classes

    if args.dataset == 'ImageNet':
        from utils import load_imagenet_folder2name
        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])

        class_names = new_class_names

    # Original class name
    class_names = refine_classname(class_names)
    # Context + Class name
    texts_train = [template.format(label) for label in class_names]


    ###### Save the original classnames for Text Prompt Tuning
    training_original_classnames = class_names

    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts
        else:
            class_names = each.classes
            if val_dataset_name[cnt] == 'ImageNet':
                from utils import load_imagenet_folder2name
                folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                new_class_names = []
                for class_name in class_names:
                    new_class_names.append(folder2name[class_name])
                class_names = new_class_names

            class_names = refine_classname(class_names)
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)

    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)


    args.model_folder = os.path.join(args.model_dir, args.filename)
    if dist.get_rank() == 0:
        if not os.path.isdir(args.model_folder):
            os.makedirs(args.model_folder)

    # wandb
    # if args.use_wandb:
    #     wandb.init(project='Visual Prompting')
    #     wandb.config.update(args)
    #     wandb.run.name = args.filename
    #     wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model,
                             prompter, add_prompter, criterion, args)
        return

    epochs_since_improvement = 0

    #################################### Constructing Text Prompter ####################################
    prompt_learner = None
    if args.adv_prompt_gen:
        prompt_learner = PromptLearner(args, training_original_classnames, model).to(device)
        # prompt_learner = torch.nn.DataParallel(prompt_learner).cuda()
        if args.arch == 'RN50' or args.arch == 'RN101':
            if dist.get_world_size() > 1:
                prompt_learner = nn.SyncBatchNorm.convert_sync_batchnorm(prompt_learner)
            prompt_learner = DDP(prompt_learner, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        else:
            prompt_learner = DDP(prompt_learner, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # prompter_optim = torch.optim.SGD(prompt_learner,
        #                                  lr=args.text_perb_stepsize,
        #                                  momentum=0,
        #                                  weight_decay=0)
    
    #################################### Constructing Text Prompter ####################################


    for epoch in range(args.start_epoch, args.epochs):
        dist.barrier()
        train(train_loader, texts_train, model, original_model, prompter, add_prompter, 
              optimizer, scheduler, criterion, scaler, epoch, prompt_learner, args)
        
        ### SWA ###
        if epoch == args.swa_start:
            swa_model.module.visual.load_state_dict(model.module.visual.state_dict())
        elif epoch > args.swa_start:
            moving_average(swa_model, model, args.swa_decay)
            bn_update(train_loader, swa_model)
        ### SWA ###

        # evaluate on validation set
        if epoch % args.validate_freq == 0:
            if dist.get_rank() == 0:
                print("Validation Orginal Model:")
            acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model,
                                 prompter, add_prompter, criterion, args)
            if epoch >= args.swa_start:
                if dist.get_rank() == 0:
                    print("Validation Weight-Average Model:")
                acc1_mean_swa = validate(val_loader_list, val_dataset_name, texts_list, swa_model,
                                         prompter, add_prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        if epoch >= args.swa_start:
            better_acc1_mean = max(acc1_mean, acc1_mean_swa)
        else:
            better_acc1_mean = acc1_mean
        is_best = better_acc1_mean > best_acc1
        best_acc1 = max(better_acc1_mean, best_acc1)


        if epoch >= args.swa_start and acc1_mean_swa > acc1_mean:
            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': prompter.state_dict(),
                    'add_prompter': add_prompter.state_dict(),
                    'vision_encoder_state_dict': swa_model.module.visual.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, args, is_best=is_best)
        else:
            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': prompter.state_dict(),
                    'add_prompter': add_prompter.state_dict(),
                    'vision_encoder_state_dict': model.module.visual.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if dist.get_rank() == 0:
                print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                if dist.get_rank() == 0:
                    print("The training halted by early stopping criterion.")
                break

        
        dist.barrier()

    # wandb.run.finish()


# def validate(val_loader, texts, model, prompter, add_prompter, criterion, args):
def validate(val_loader_list, val_dataset_name, texts_list, model,
             prompter, add_prompter, criterion, args):
    dataset_num = len(val_loader_list)
    acc_all_nat = []
    acc_all_adv = []

    if args.evaluate:
        if args.eval_template == '0':
            args.val_log_name = os.path.join(args.model_folder, 'val_eps{}_{}.log'.format(args.original_test_eps, args.attack))
        else:
            args.val_log_name = os.path.join(args.model_folder, 'val_eps{}_{}_template{}.log'.format(args.original_test_eps, args.attack, args.eval_template))
        reset_log_file(args.val_log_name)
        if dist.get_rank() == 0:
            log_record(args.val_log_name, args.val_log_name)

    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):

        val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        binary = ['PCAM']
        attacks_to_run=['apgd-ce', 'apgd-dlr']
        if dataset_name in binary:
            attacks_to_run=['apgd-ce']

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')
        top1_adv_prompt = AverageMeter('Adv Prompt Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1_org, top1_adv_org],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()

        # print(val_dataset_name, 'text token', texts_list)

        #
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader, ncols = 80)):

            if 'cifar' not in val_dataset_name:
                if i % 20 != 0 and not args.evaluate:
                    continue

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            # print(target)
            # if i > 5:
            #     break
            # print("current i:", i)
            # print(images.size())

            with autocast():

                # clean images, with prompt and without prompt
                # compute output
                with torch.no_grad():
                    # prompt_token = add_prompter()
                    prompt_token = None
                    # output_prompt, _ = model(prompter(clip_img_preprocessing(images)), text_tokens, prompt_token)
                    output_prompt, _ = multiGPU_CLIP(model, prompter(clip_img_preprocessing(images)), text_tokens, prompt_token)
                    # print(output_prompt.shape)

                    loss = criterion(output_prompt, target)

                    # measure accuracy and record loss
                    acc1 = accuracy(output_prompt, target, topk=(1,))
                    losses.update(loss.item(), images.size(0))
                    # top1_prompt.update(acc1[0].item(), images.size(0))
                    top1_org.update(acc1[0].item(), images.size(0))

                # torch.cuda.empty_cache()

                # generate adv example
                if args.attack == 'CW':
                    delta_prompt = attack_CW(prompter, model, add_prompter, criterion,
                                             images, target, text_tokens,
                                             test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                    attacked_images = images + delta_prompt
                elif args.attack == 'AA':
                    attacked_images = attack_auto_new(model, images, target, text_tokens,
                        None, None, epsilon=args.test_eps, attacks_to_run=attacks_to_run)
                else:
                    delta_prompt = attack_pgd(prompter, model, add_prompter, criterion,
                                              images, target, text_tokens,
                                              test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                    attacked_images = images + delta_prompt

                # compute output
                torch.cuda.empty_cache()
                dist.barrier()
                with torch.no_grad():
                    prompt_token = add_prompter()
                    # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)
                    output_prompt_adv, _ = multiGPU_CLIP(model, prompter(clip_img_preprocessing(attacked_images)), text_tokens, prompt_token)

                    loss = criterion(output_prompt_adv, target)

                # bl attack
                torch.cuda.empty_cache()
                dist.barrier()

                # measure accuracy and record loss
                acc1 = accuracy(output_prompt_adv, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_adv_org.update(acc1[0].item(), images.size(0))
                # top1_adv_prompt.update(acc1[0].item(), images.size(0))

                # acc1 = accuracy(output_org_adv, target, topk=(1,))
                # top1_adv_org.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0  and i != 0:
                if dist.get_rank() == 0:
                    progress.display(i)

        torch.cuda.empty_cache()
        dist.barrier()

        # print(dataset_name + ' * Adv Prompt Acc@1 {top1_adv_prompt.avg:.3f} Adv Original Acc@1 {top1_adv_org.avg:.3f} '
        #                      '*  Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
        #       .format(top1_adv_prompt=top1_adv_prompt, top1_adv_org=top1_adv_org,
        #               top1_prompt=top1_prompt, top1_org=top1_org))
        if args.evaluate:
            if dist.get_rank() == 0:
                log_record(dataset_name + '--- Clean Acc.: {top1_org.avg:.2f}  Adv Acc.: {top1_adv_org.avg:.2f}.'
                    .format(top1_org=top1_org, top1_adv_org=top1_adv_org), args.val_log_name)
        else:
            if dist.get_rank() == 0:
                print(dataset_name + '--- Clean Acc.: {top1_org.avg:.2f}  Adv Acc.: {top1_adv_org.avg:.2f}.'
                    .format(top1_org=top1_org, top1_adv_org=top1_adv_org))

        acc_all_nat.append(top1_org.avg)
        acc_all_adv.append(top1_adv_org.avg)

    # if args.use_wandb:
    #     wandb.log({
    #         'val_loss': losses.avg,
    #         'val_acc_prompt': top1_prompt.avg,
    #         'val_acc_org': top1_org.avg,
    #     })
    if args.evaluate:
        if dist.get_rank() == 0:
            log_record('Average on all datasets --- Clean Acc.: {:.2f}  Adv Acc.: {:.2f}.'
                    .format(np.mean(acc_all_nat), np.mean(acc_all_adv)), args.val_log_name)
    else:
        if dist.get_rank() == 0:
            print('Average on all datasets --- Clean Acc.: {:.2f}  Adv Acc.: {:.2f}.'
                    .format(np.mean(acc_all_nat), np.mean(acc_all_adv)))
    
    return np.mean(acc_all_adv)


if __name__ == '__main__':
    main()
