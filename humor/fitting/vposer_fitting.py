# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.02.12


import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))


from datasets.ceti_fit_dataset import CeTIFitDataset
from datasets.horst_fit_dataset import HorstFitDataset

from typing import Union

import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from torch import nn

import numpy as np
import os

from os import listdir
from os.path import isfile, join
import copy

os.environ['PYOPENGL_PLATFORM'] = 'egl'

#from human_body_prior.models.ik_engine import IK_Engine
import human_body_prior.models.ik_engine
from importlib import reload
from os import path as osp
import os
from human_body_prior.tools.omni_tools import create_list_chunks

from anonymization.anon_viz import render_pose_seq


#from body_visualizer.tools.vis_tools import render_smpl_params
#from body_visualizer.tools.vis_tools import imagearray2file
#from body_visualizer.tools.vis_tools import show_image

CETI_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 4, 'spine' : None, 'leftLeg' : 2, 'rightLeg' : 5,
                'spine1' : None, 'leftFoot' : 3, 'rightFoot' : 6, 'spine2' : 7, 'leftToeBase' : None, 'rightToeBase' : None,
                'neck' : None, 'leftShoulder' : 8, 'rightShoulder' : 13, 'head' : 12, 'leftArm' : 9, 'rightArm' : 14,
                'leftForeArm' : 10, 'rightForeArm' : 15, 'leftHand' : 11, 'rightHand' : 16}


SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11,
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}

# The SMPL JOINT Names in the correct order
SMPL_JOINTS_NAMES = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg', 'spine1', 'leftFoot',
                     'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase', 'neck', 'leftShoulder', 'rightShoulder',
                     'head', 'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand']



def fit_sequence(sequence, selected_joints):
    data_loss = torch.nn.MSELoss(reduction='sum')

    stepwise_weights = [
        {'data': 10., 'poZ_body': .01, 'betas': .5},
    ]

    optimizer_args = {'type': 'LBFGS', 'max_iter': 300, 'lr': 1, 'tolerance_change': 1e-4, 'history_size': 200}
    ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                          verbosity=2,
                          display_rc=(2, 2),
                          data_loss=data_loss,
                          stepwise_weights=stepwise_weights,
                          num_betas=10,
                          optimizer_args=optimizer_args).to(comp_device)

    frame_ids = np.arange(len(sequence))
    batch_size = 20

    target_poses_new = []
    poses_new = []
    ik_results = {"trans": [], "betas": [], "root_orient": [], "poZ_body": [], "pose_body": []}
    for rnd_frame_ids in create_list_chunks(frame_ids, batch_size, overlap_size=0, cut_smaller_batches=False):
        if len(rnd_frame_ids) < batch_size:
            break

        #print(rnd_frame_ids)

        target_pts = sequence[rnd_frame_ids, :].detach().to(comp_device)
        source_pts = SourceKeyPoints(bm=bm_fname, n_joints=n_joints, selected_joints=selected_joints).to(comp_device)

        ik_res = ik_engine(source_pts, target_pts)

        new_body = BodyModel(bm_fname).to(comp_device)
        new_body = new_body(**ik_res)
        output_joint_pos = torch.index_select(new_body.Jtr, 1, selected_joints).detach().cpu().numpy()
        poses_new.append(output_joint_pos)

        ik_res_detached = {k: v.detach().cpu() for k, v in ik_res.items()}

        for ele in ik_res_detached:
            ik_results[ele].append(ik_res_detached[ele])

        target_poses_new.append(target_pts.cpu().numpy())

        nan_mask = torch.isnan(ik_res_detached['trans']).sum(-1) != 0
        if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')



    if len(poses_new) > 0:
        for ele in ik_results:
            ik_results[ele] = np.concatenate(ik_results[ele])
        return {"positions": np.concatenate(poses_new), "ik_results": ik_results}
    return None


def fit_sequence_full(sequence, selected_joints):

    data_loss = torch.nn.MSELoss(reduction='sum')

    stepwise_weights = [
        {'data': 10., 'poZ_body': .01, 'betas': .5},
    ]

    # This model retains some states due to closures, we reload it to reset it.
    reload(human_body_prior.models.ik_engine)

    optimizer_args = {'type': 'LBFGS', 'max_iter': 300, 'lr': 1, 'tolerance_change': 1e-4, 'history_size': 200}
    ik_engine = human_body_prior.models.ik_engine.IK_Engine(vposer_expr_dir=vposer_expr_dir,
                          verbosity=2,
                          display_rc=(2, 2),
                          data_loss=data_loss,
                          stepwise_weights=stepwise_weights,
                          num_betas=10,
                          optimizer_args=optimizer_args).to(comp_device)

    ik_results = {}

    target_pts = sequence.detach().to(comp_device)
    source_pts = SourceKeyPoints(bm=bm_fname, n_joints=22, selected_joints=selected_joints).to(comp_device)

    ik_res = ik_engine(source_pts, target_pts)

    new_body = BodyModel(bm_fname).to(comp_device)
    new_body = new_body(**ik_res)
    if selected_joints:
        output_joint_pos = torch.index_select(new_body.Jtr, 1, selected_joints).detach().cpu().numpy()
    else:
        output_joint_pos = new_body.Jtr.detach().cpu().numpy()[:,:22]
    output_joint_pos_smpl = new_body.Jtr[:,:21].detach().cpu().numpy()

    ik_res_detached = {k: v.detach().cpu() for k, v in ik_res.items()}

    for ele in ik_res_detached:
        ik_results[ele] = ik_res_detached[ele].numpy()

    nan_mask = torch.isnan(ik_res_detached['trans']).sum(-1) != 0
    if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')


    for ele in ik_results:
        ik_results[ele] = np.array(ik_results[ele])
    return {"positions": output_joint_pos, "positions_all": output_joint_pos_smpl, "ik_results": ik_results}





class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 n_joints: int=22,
                 selected_joints: Union[torch.tensor, None] = None,
                 kpts_colors: Union[np.ndarray, None] = None ,
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []#self.bm.f
        self.n_joints = n_joints
        self.selected_joints = selected_joints
        self.kpts_colors = np.array([Color('grey').rgb for _ in range(n_joints)]) if kpts_colors == None else kpts_colors

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        if selected_joints:
            source_kpts = torch.index_select(new_body.Jtr, 1, self.selected_joints)
        else:
            source_kpts = new_body.Jtr[:,:self.n_joints]

        return {'source_kpts':source_kpts, 'body': new_body}




support_dir = '../human_body_prior/support_data/dowloads'
data_import_dir = join('../data/original/Horst_Study/')
output_dir = '../human_body_prior/support_data/data/Horst_fitted'
vposer_expr_dir = osp.join(support_dir,'vposer_v2_05') #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
bm_fname =  osp.join(support_dir,'models/smplx/neutral/model.npz')#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads

out_dir = "./"


#comp_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
comp_device = torch.device('cpu')

#dataset = CeTIFitDataset(data_import_dir, return_joints=True, return_points=True, split_size=30, split_part=0)
dataset = HorstFitDataset(data_import_dir, return_joints=True, return_points=True)

#selected_joints = CeTIFitDataset.SMPL_joint_selection_mask().to(comp_device)
#n_joints = len(selected_joints)
selected_joints = None
n_joints = 22

already_fitted = [f for f in listdir(output_dir) if isfile(join(output_dir, f))]
already_fitted = [f for f in already_fitted if "motion.tsv.npz" in f]

for i, sample in enumerate(dataset):
    observed_dict, _ = sample

    if observed_dict["meta"]["file_path"] + ".npz" in already_fitted:
        print("Already fitted, skipping")
        continue

    sequence = torch.tensor(observed_dict["joints3d"]).float().to(comp_device)
    #sequence = torch.index_select(sequence, 1, selected_joints)

    fitted_sequence = fit_sequence_full(sequence, selected_joints)
    if fitted_sequence is None:
        print("Empty fitted sequence")
        continue

    print(fitted_sequence["positions"].shape)

    #original_seq = sequence.reshape(-1, n_joints * 3)
    #fitted_seq = fitted_sequence["positions"].reshape(-1, n_joints * 3)
    #print(original_seq.shape)
    #print(fitted_seq.shape)
    #render_pose_seq([original_seq, fitted_seq], osp.join(out_dir, "file_points_cmp.gif"))

    np.savez(osp.join(output_dir, observed_dict["meta"]["file_path"]), positions=fitted_sequence["positions"],
             positions_all=fitted_sequence["positions_all"],
             trans=fitted_sequence["ik_results"]["trans"],
             betas=fitted_sequence["ik_results"]["betas"],
             root_orient=fitted_sequence["ik_results"]["root_orient"],
             poZ_body=fitted_sequence["ik_results"]["poZ_body"],
             pose_body=fitted_sequence["ik_results"]["pose_body"])
