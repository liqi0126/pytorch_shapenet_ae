"""
    For models: model_v1_x.py
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from PIL import Image
from subprocess import call
from data import PartNetShapeDataset
import utils
from geometry_utils import render_pts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


### get parameters
parser = ArgumentParser()

# main parameters (required)
parser.add_argument('--exp_name', type=str, help='exp name')
parser.add_argument('--model_version', type=str, help='model def file', default='model_v3')

# main parameters (optional)
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
#parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')

# network settings
parser.add_argument('--model_epoch', type=int, default=-1)
parser.add_argument('--num_test', type=int, default=10)
parser.add_argument('--interp_step', type=int, default=10)

# parse args
conf = parser.parse_args()
conf.flog = None

# load train config
train_conf = torch.load(os.path.join(conf.log_dir, conf.exp_name, 'conf.pth'))

# load network model
model_def = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {device}')

# find the model_epoch
if conf.model_epoch < 0:
    for item in os.listdir(os.path.join(conf.log_dir, conf.exp_name, 'ckpts')):
        if '_net_network.pth' in item:
            conf.model_epoch = max(int(item.split('_')[0]), conf.model_epoch)

# check if eval results already exist. If so, delete it.
result_dir = os.path.join('logs', conf.exp_name, f'interp-model_epoch_{conf.model_epoch}')
if os.path.exists(result_dir):
    if not conf.overwrite:
        response = input('Eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
        if response != 'y':
            sys.exit()
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
print(f'\nTesting under directory: {result_dir}\n')

# create models
network = model_def.Network(train_conf)
utils.printout(conf.flog, '\n' + str(network) + '\n')

# load pretrained model
print('Loading ckpt from ', os.path.join(conf.log_dir, conf.exp_name, 'ckpts'), conf.model_epoch)
data_to_restore = torch.load(os.path.join('logs', conf.exp_name, 'ckpts', '%d_net_network.pth' % conf.model_epoch))
network.load_state_dict(data_to_restore, strict=False)
print('DONE\n')

# send parameters to device
network.to(device)

# set eval mode
network.eval()

# prepare visu subfolders
visu_subdirs = []; visu_subdirs_str = '';
for i in range(conf.interp_step):
    cur_visu_subdir = os.path.join(result_dir, 'pc-%d'%i)
    if i == 0:
        visu_subdirs_str += 'pc-%d'%i
    else:
        visu_subdirs_str += ',pc-%d'%i
    os.mkdir(cur_visu_subdir)
    visu_subdirs.append(cur_visu_subdir)


# main
with torch.no_grad():
    for i in range(conf.num_test):
        print('%d/%d ...' % (i, conf.num_test))

        z1 = torch.randn(1, 128).to(device).repeat(conf.interp_step, 1)
        z2 = torch.randn(1, 128).to(device).repeat(conf.interp_step, 1)
        alpha = torch.arange(conf.interp_step).to(device).float() / (conf.interp_step - 1)
        alpha = alpha.unsqueeze(1).repeat(1, 128)
        zs = z1 * alpha + z2 * (1 - alpha)

        # forward through the network
        output_pcs = network.infer(zs)

        # visu
        for j in range(conf.interp_step):
            out_fn = os.path.join(visu_subdirs[j], 'test-%03d.png'%i)
            render_pts(out_fn, output_pcs[j].cpu().numpy())
                
    # visu html
    utils.printout(conf.flog, 'Generating html visualization ...')
    cmd = 'cd %s && python %s . 10 htmls %s %s > /dev/null' % (result_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierarchy_local.py'), visu_subdirs_str, visu_subdirs_str)
    call(cmd, shell=True)
    utils.printout(conf.flog, 'DONE')

