import os, sys
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import json
import yaml
import numpy as np
import random
from tqdm import tqdm
import torch
from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS
from anonymization.anon_viz import render_pose_seq


def get_SMPL_joint_positions(data):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    body_pose = torch.tensor(data["pose_body"]).to(device).type(torch.float)
    trans = torch.tensor(data["trans"]).to(device).type(torch.float)
    root_orient = torch.tensor(data["root_orient"]).to(device).type(torch.float)
    betas = torch.tensor(data["betas"]).to(device).type(torch.float)

    if len(betas.size()) == 2:
        num_betas = betas.size()[1]
    else:
        num_betas = betas.size()[0]

    bm = BodyModel(bm_path="./body_models/smplh/neutral/model.npz",
                                num_betas=num_betas,
                                batch_size=body_pose.size()[0],
                                use_vtx_selector=True, model_type='smplh').to(device)

    if len(betas.size()) != len(body_pose.size()):
        betas = betas.expand((body_pose.size()[0], -1))

    smpl_body = bm(pose_body=body_pose,
                                pose_hand=None,
                                betas=betas,
                                root_orient=root_orient
                                ) #trans=trans

    # body joints
    joints3d = smpl_body.Jtr[:, :len(SMPL_JOINTS), :]
    joints3d = joints3d.reshape(-1, len(SMPL_JOINTS) * 3)
    return joints3d


def process_sequence(in_path, out_path, participant, action, action_full, sequence):

    path = in_path + sequence

    data =  np.load(path)

    tmp_body_pose = data["body_pose"].reshape(-1, 69)

    tmp = {"trans": data["transl"], "root_orient": data["global_orient"],
           "pose_body": tmp_body_pose[:,:63], "betas": data["betas"]}

    tmp["positions"] = get_SMPL_joint_positions(tmp).detach().numpy()

    meta_data = {"participant_id": participant, "modality": action}

    file_full_path = out_path + participant + "." + action_full
    np.savez(file_full_path + ".npz", **tmp)

    with open(file_full_path + '.yaml', 'w') as file:
        yaml.dump(meta_data, file)

    return tmp


def get_filtered_files(in_path):

    files = os.listdir(in_path)

    participants = {}
    for file in tqdm(files):
        participant = file.split('_')[0]
        action = file.split('_')[1][1:-4]

        if participant not in participants:
            participants[participant] = [action]
        else:
            participants[participant].append(action)

    filtered_participants = {}
    for participant in participants:
        if len(participants[participant]) >= 40:
            filtered_participants[participant] = participants[participant]

    actions = {}
    for participant in filtered_participants:
        for a in filtered_participants[participant]:
            if a not in actions:
                actions[a] = [participant]
            else:
                actions[a].append(participant)

    filtered_actions = {}
    for action in actions:
        if len(actions[action]) >= 10:
            filtered_actions[action] = actions[action]

    filtered_files = []
    for file in files:
        participant = file.split('_')[0]
        action = file.split('_')[1][1:-4]

        if participant in filtered_participants and action in filtered_actions:
            filtered_files.append(file)

    return filtered_files



def process_folder(files, in_path, out_path):
    #files = os.listdir(in_path)

    participants = []
    for file in tqdm(files):
        participant = file.split('_')[0]
        participants.append(participant)
        action = file.split('_')[1][1:-4]
        action_full = file.split('_')[1][:-4]
        process_sequence(in_path, out_path, participant, action, action_full, file)

    for participant in set(participants):
        meta_data = {"participant_id": participant}
        with open(out_path + participant + '.yaml', 'w') as file:
            yaml.dump(meta_data, file)



if __name__ == '__main__':

    in_path = "../data/original/HuMMan/data/smpl_on_ground/"
    out_path = "../data/prepared/HuMMan/"

    files = get_filtered_files(in_path)


    process_folder(files, in_path, out_path)





