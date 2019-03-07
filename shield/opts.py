import os

import numpy as np
from cleverhans.attacks import FastGradientMethod, DeepFool, CarliniWagnerL2

from constants import CHECKPOINTS_DIR
from models.inception_v4 import InceptionV4
from models.resnet_50_v2 import ResNet50v2
from utils.defenses import (jpeg_compress, slq,
                            median_filter,
                            denoise_tv_bregman)


# -----

models = ['resnet_50_v2', 'inception_v4']
attacks = ['fgsm', 'df', 'cwl2']
defenses = ['jpeg', 'slq', 'median_filter', 'tv_bregman']
tf_defenses = ['jpeg', 'slq']

# -----

model_class_map = {
    'resnet_50_v2': ResNet50v2,
    'inception_v4': InceptionV4}

model_checkpoint_map = {
    'resnet_50_v2': os.path.join(CHECKPOINTS_DIR, 'resnet_v2_50.ckpt'),
    'inception_v4': os.path.join(CHECKPOINTS_DIR, 'inception_v4.ckpt')}

model_slim_name_map = {
    'resnet_50_v2': 'resnet_v2_50',
    'inception_v4': 'inception_v4'}

# -----

attack_class_map = {
    'fgsm': FastGradientMethod,
    'df': DeepFool,
    'cwl2': CarliniWagnerL2}

attack_options_map = {
    'fgsm': {
        'ord': np.inf,
        'eps': (2. * 8. / 255.),
        'clip_min': -1., 'clip_max': 1.},
    'df': {
        'nb_candidate': 10,
        'max_iter': 100,
        'clip_min': -1., 'clip_max': 1.},
    'cwl2': {
        'confidence': 0,
        'learning_rate': 5e-3,
        'batch_size': 4,
        'clip_min': -1., 'clip_max': 1.}}

# -----

defense_fn_map = {
    'jpeg': jpeg_compress,
    'slq': slq,
    'median_filter': median_filter,
    'tv_bregman': denoise_tv_bregman}

defense_options_map = {
    'jpeg': {'quality': 80},
    'slq': {},
    'median_filter': {'size': 3},
    'tv_bregman': {'weight': 30}}
