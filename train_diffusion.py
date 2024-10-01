import argparse, wandb, logging, torch, os, json, random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from custom_pytorch_utils.custom_transforms import *
from pytorch_models.BphPSEG import BphPSEG
from pytorch_models.BphPQUANT import BphPQUANT
from pytorch_models.MLP import inherit_mlp_class_from_parent
from pytorch_models.Unet import inherit_unet_pretrained_class_from_parent
from pytorch_models.BphP_deeplabv3 import inherit_deeplabv3_smp_resnet101_class_from_parent
from pytorch_models.BphP_segformer import inherit_segformer_class_from_parent
from custom_pytorch_utils.custom_transforms import *
from custom_pytorch_utils.custom_datasets import *
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import custom_pytorch_utils.augment_models_func as amf