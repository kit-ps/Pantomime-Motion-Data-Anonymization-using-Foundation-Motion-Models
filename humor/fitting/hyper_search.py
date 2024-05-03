
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from fitting.config import parse_args
from fitting.run_fitting_pantomime import main

from datasets.horst_fit_dataset import  HorstFitDataset
from datasets.ceti_fit_dataset import CeTIFitDataset

from anonymization.prepare_data import load_data_vposer_humor_fitted, load_data_humor_fitted, get_SMPL_joint_positions

from torch.multiprocessing import Pool, set_start_method

import optuna
import shutil
import torch
import numpy as np
from os import path as osp
from os import listdir


# fix random seeds for reproducibility
SEED = 234234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def calculate_fit_quality(data_path_original, data_path_fit, dataset_name):


    pred_dict_humor = load_data_humor_fitted(data_path_fit + data_path_original[2:])
    pred_dict_vposer = load_data_vposer_humor_fitted(data_path_fit + data_path_original[2:])

    if dataset_name == "CeTI":
        grnd_truth_dataset = CeTIFitDataset(data_path_original, return_joints=True)
    else:
        grnd_truth_dataset = HorstFitDataset(data_path_original, return_joints=True)

    grnd_truth_dict = {}

    for i, data in enumerate(grnd_truth_dataset):
        observed_dict, gt_dict = data
        key = observed_dict["meta"]["file_path"].split(".")[0]
        if dataset_name == "CeTI":
            # Removes the _motion suffix from the key
            key = key[:-7]
        else:
            # Removes the _Gait suffix from the key
            key = key[:-5]
        grnd_truth_dict[key] = observed_dict["joints3d"]

    # removes the random number from the end of the folder name
    tmp_humor, tmp_vposer = {}, {}
    for ele in pred_dict_humor:
        tmp = "_".join(ele.split("_")[:-1])
        tmp_humor[tmp] = pred_dict_humor[ele]
        tmp_vposer[tmp] = pred_dict_vposer[ele]
    pred_dict_vposer = tmp_vposer
    pred_dict_humor = tmp_humor

    keys_grd = list(grnd_truth_dict.keys())
    keys_humor = list(pred_dict_humor.keys())

    print("grnd", keys_grd)
    print("humor", keys_humor)

    # Get the pose indices which are not inf
    filter_indices = []
    for i in range(len(grnd_truth_dict[keys_grd[0]][0])):
        ele = grnd_truth_dict[keys_grd[0]][0][i]
        if float("inf") not in ele:
            filter_indices.append(i)

    results = {"humor": [], "vposer": []}
    for key in keys_humor:
        ground_truth_data = grnd_truth_dict[key]
        ground_truth_data = np.take(ground_truth_data, filter_indices, 1)
        ground_truth_data = ground_truth_data.reshape(ground_truth_data.shape[0], -1)
        predicted_data_humor = get_SMPL_joint_positions(pred_dict_humor[key]).detach().cpu().numpy().reshape(-1, 22, 3)
        predicted_data_humor = np.take(predicted_data_humor, filter_indices, 1)
        predicted_data_humor = predicted_data_humor.reshape(predicted_data_humor.shape[0], -1)
        predicted_data_vposer = get_SMPL_joint_positions(pred_dict_vposer[key]).detach().cpu().numpy().reshape(-1, 22, 3)
        predicted_data_vposer = np.take(predicted_data_vposer, filter_indices, 1)
        predicted_data_vposer = predicted_data_vposer.reshape(predicted_data_vposer.shape[0], -1)

        results["humor"].append(np.linalg.norm(predicted_data_humor - ground_truth_data, axis=1))
        results["vposer"].append(np.linalg.norm(predicted_data_vposer - ground_truth_data, axis=1))

    ade_res = {}
    for fit in ["humor", "vposer"]:
        fde = 0
        ade = 0
        for ele in results[fit]:
            fde += ele[-1]
            ade += ele.mean()

        fde /= len(results[fit])
        ade /= len(results[fit])

        ade_res[fit] = ade

    return ade_res["humor"], ade_res["vposer"]



# Mutes stdout
def mute():
    sys.stdout = open(os.devnull, 'w')

def run_fitting(args, config_file):

    num_pools = 10
    tmp = [(args, config_file, num_pools, i) for i in range(num_pools)]

    # Pytorch foo that allows multitasking with
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    with Pool(num_pools, initializer=mute) as p:
        p.map(main, tmp)

    result = calculate_fit_quality(args.data_path, args.out, args.data_type)

    return result


def draw_parameters(trial, args):

    args.joint3d_weight = [trial.suggest_float('joint3d_weight_one', 0.0, 20),
                           trial.suggest_float('joint3d_weight_two', 0.0, 20),
                           trial.suggest_float('joint3d_weight_three', 0.0, 20)]
    args.pose_prior_weight = [trial.suggest_float('pose_prior_weight_one', 0.0, 1.0),
                              trial.suggest_float('pose_prior_weight_two', 0.0, 1.0),
                              0.0]
    args.shape_prior_weight = [trial.suggest_float('shape_prior_weight_one', 0.0, 1.0),
                               trial.suggest_float('shape_prior_weight_two', 0.0, 1.0),
                               trial.suggest_float('shape_prior_weight_three', 0.0, 1.0)]
    args.motion_prior_weight = [0.0, 0.0, trial.suggest_float('motion_prior_weight_three', 0.0, 1.0)]
    args.init_motion_prior_weight = [0.0, 0.0, trial.suggest_float('init_motion_prior_weight_three', 0.0, 1.0)]
    args.joint3d_smooth_weight = [trial.suggest_float('joint3d_smooth_weight_one', 0.0, 1.0),
                                  trial.suggest_float('joint3d_smooth_weight_two', 0.0, 1.0),
                                  0.0]
    args.joint_consistency_weight = [0.0, 0.0, trial.suggest_float('joint_consistency_weight_three', 0.0, 10.0)]
    args.bone_length_weight  = [0.0, 0.0, trial.suggest_float('bone_length_weight_three', 0.0, 20.0)]

    """
    joint3d-weight 10.0 10.0 10.0
    --pose-prior-weight 2e-4 2e-4 0.0
    --shape-prior-weight 1.67e-4 1.67e-4 1.67e-4
    
    --motion-prior-weight 0.0 0.0 1e-3
    
    --init-motion-prior-weight 0.0 0.0 1e-3
    
    --joint3d-smooth-weight 0.1 0.1 0.0
    
    --joint-consistency-weight 0.0 0.0 1.0
    --bone-length-weight 0.0 0.0 10.0
    """



# 1. Define an objective function to be maximized.
def objective(trial):
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]

    shutil.rmtree(args.out, ignore_errors=True)

    draw_parameters(trial, args)

    result = run_fitting(args, config_file)

    return result

if __name__ == '__main__':

    # 3. Create a study object and optimize the objective function.
    study = optuna.create_study(storage="sqlite:///hyper_search.db", directions=['minimize', 'minimize'])
    study.optimize(objective, n_trials=100)