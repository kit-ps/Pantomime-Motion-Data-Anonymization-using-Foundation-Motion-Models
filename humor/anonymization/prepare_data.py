import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import csv
import sys
import os
import copy
import torch
import yaml
import numpy as np
from os import path as osp
from os import listdir
from pathlib import Path

from anonymization.motion_anonymization import MotionAnonymization

from fitting.fitting_utils import load_vposer

from body_model.utils import SMPL_JOINTS

from body_model.body_model import BodyModel

from anonymization.anon_util import format_positions_as_original

from datasets.horst_fit_dataset import HorstFitDataset

J_BODY = len(SMPL_JOINTS)-1 # no root


def prepare_ceti_original_sequence(sequence):
    sequence = resample_sequence(sequence, 100, 30)
    prep_seq = dict()
    prep_seq["positions"] = sequence[:, :17 * 3]
    prep_seq["rotations"] = sequence[:, 17 * 3:]

    return prep_seq


def prepare_vposers_fitted_sequence(sequence):
    for ele in sequence:
        sequence[ele] = sequence[ele].reshape(sequence[ele].shape[0], -1)

    sequence["joint_positions_smpl"] = sequence.pop("positions_all")

    return sequence


def infer_humor_latent_motion(data):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    input = dict()

    input["pose_body"] = torch.tensor(data["pose_body"].reshape(1, -1, 63)).to(device).type(torch.float)
    input["trans"] = torch.tensor(data["trans"].reshape(1, -1, 3)).to(device).type(torch.float)
    input["root_orient"] = torch.tensor(data["root_orient"].reshape(1, -1, 3)).to(device).type(torch.float)
    input["betas"] = torch.tensor(data["betas"][0].reshape(1, -1)).to(device).type(torch.float)

    args = {
        "smpl": "./body_models/smplh/neutral/model.npz",
        "out": None,
        "humor": "./checkpoints/humor/best_model.pth",
        "vposer": "./body_models/vposer_v1_0",
        "pose_by_pose": False,
        "humor_in_rot_rep": "mat",
        "humor_out_rot_rep": "aa",
        "humor_latent_size": 48,
        "humor_model_data_config": "smpl+joints+contacts",
        "humor_steps_in": 1,
        "batch_size": 1
    }
    # Makeshift object as this is expected by the MotionAnonymization
    ObjFromDict = type('ObjFromDict', (object,), args)
    args = ObjFromDict()

    # Blocks the prints in the Class Init
    sys.stdout = open(os.devnull, 'w')
    motion_anon = MotionAnonymization(device, args, 1, "normal", num_betas=input["betas"].size()[1], batch_size=1,
                                      seq_len=input["pose_body"].size()[1])
    sys.stdout = sys.__stdout__

    latent_motion, prep_seq = motion_anon.infer_latent_motion(input["trans"], input["root_orient"], input["pose_body"],
                                                               input["betas"], data_fps=30, full_forward_pass=False)

    return latent_motion.reshape(-1, args.humor_latent_size)


def infer_vposer_latent_pose(data):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # Blocks the prints in the function
    sys.stdout = open(os.devnull, 'w')
    pose_prior, _ = load_vposer("./body_models/vposer_v1_0")
    sys.stdout = sys.__stdout__
    pose_prior = pose_prior.to(device)

    body_pose = torch.tensor(data["pose_body"]).to(device)
    latent_pose_distrib = pose_prior.encode(body_pose.type(torch.float))
    latent_pose = latent_pose_distrib.mean.reshape((-1 , pose_prior.latentD))

    return latent_pose


def produce_all_position_combinations(sample, dataset_name):
    combinations = [["trans", "betas", "root_orient"], ["trans", "betas", "pose_body"],
                    ["trans", "root_orient", "pose_body"], ["betas", "root_orient", "pose_body"],
                    ["betas"], ["root_orient"], ["pose_body"], ["trans"]]
    sample["positions"] = get_SMPL_joint_positions(sample)

    for comb in combinations:
        tmp = {}

        for a in ["trans", "betas", "root_orient", "pose_body"]:
            if a in comb:
                tmp[a] = np.zeros(sample[a].shape)
            else:
                tmp[a] = sample[a]

        name = "positions_without"
        for ele in comb:
            name += "_" + ele

        sample[name] = get_SMPL_joint_positions(tmp)

    return sample


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
                                root_orient=root_orient,
                                trans=trans)

    # body joints
    joints3d = smpl_body.Jtr[:, :len(SMPL_JOINTS), :]
    joints3d = joints3d.reshape(-1, len(SMPL_JOINTS) * 3)
    return joints3d


def resample_sequence(sequence, original_fps, target_fps=30):
    new_data = []
    for i in range(len(sequence[0])):
        dim = sequence[:, i]

        old_times = np.arange(len(dim))
        step = original_fps / target_fps
        new_times = np.arange(old_times[0], old_times[-1], step)

        new_dim = np.interp(new_times, old_times, dim)
        new_data.append(new_dim)

    # Transpose as now as list of the dimensions, not of the frames
    new_data = np.array(new_data).T
    return new_data



def load_metadata(in_path):
    with open(osp.join(in_path, 'participants.tsv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        metadata = {}
        for row in reader:
            metadata[row["participant_id"]] = row

        return metadata


def load_data_vposer_fitted(in_path):
    samples = {}

    files = [f for f in listdir(in_path) if osp.isfile(osp.join(in_path, f))]
    motion_files = [f for f in files if ".npz" in f]

    for file in motion_files:
        file_name = file.split(".")[0]

        with np.load(osp.join(in_path, file), allow_pickle=True) as data:
            samples[file_name] = dict(data)
            samples[file_name] = prepare_vposers_fitted_sequence(samples[file_name])

    return samples

def load_data_vposer_humor_fitted(in_path):
    samples = {}

    folders = [f for f in listdir(in_path) if not osp.isfile(osp.join(in_path, f))]
    motion_folders = [f for f in folders if "tsv" in f]

    for folder in motion_folders:
        file_name = folder.split(".")[0]
        file_path = osp.join(in_path, folder, "stage2_results.npz")

        if not osp.isfile(file_path):
            print("Missing stage 2 results for: " + file_name)
            continue

        with np.load(osp.join(in_path, folder, "stage2_results.npz"), allow_pickle=True) as data:
            samples[file_name] = dict(data)

    return samples


def load_data_humor_fitted(in_path):
    samples = {}

    folders = [f for f in listdir(in_path) if not osp.isfile(osp.join(in_path, f))]
    motion_folders = [f for f in folders if "tsv" in f]

    for folder in motion_folders:
        file_name = folder.split(".")[0]
        file_path = osp.join(in_path, folder, "stage3_results.npz")

        if not osp.isfile(file_path):
            print("Missing stage 3 results for: " + file_name)
            continue

        with np.load(osp.join(in_path, folder, "stage3_results.npz"), allow_pickle=True) as data:
            samples[file_name] = dict(data)

    return samples


def load_data_ceti_original(in_path, metadata):
    samples = {}
    samples_meta = {}
    for participant in metadata:

        participant_path = osp.join(in_path, participant, "motion")
        files = [f for f in listdir(participant_path) if osp.isfile(osp.join(participant_path, f))]
        motion_files = [f for f in files if "_motion.tsv" in f]

        for file in motion_files:
            file_name = file.split(".")[0]
            with open(osp.join(participant_path, file)) as tsv_file:
                reader = csv.reader(tsv_file, delimiter='\t', quotechar='"')
                samples[file_name] = np.array(list(reader)[1:], dtype=np.float32)
                samples[file_name] = prepare_ceti_original_sequence(samples[file_name])
                samples_meta[file_name] = copy.deepcopy(metadata[participant])
                samples_meta[file_name]["modality"] = file.split("_")[1]

    return samples, samples_meta


def create_prepared_data(original_sample, vposer_sample, humor_sample):
    data = dict()

    if torch.is_tensor(original_sample):
        data["original_positions"] = original_sample.reshape(original_sample.shape[0], -1)
    else:
        data["original_positions"] = original_sample["positions"]
        data["original_rotations"] = original_sample["rotations"]


    for ele in vposer_sample:
        data["vposer_"+ele] = vposer_sample[ele]

    for ele in humor_sample:
        data["humor_"+ele] = humor_sample[ele]

    return data


def main(in_path, dataset_name, fittings):
    original_in_path = osp.join(in_path, "original", dataset_name)
    out_path = osp.join(in_path, "prepared", dataset_name)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    if dataset_name == "CeTI-Locomotion":
        metadata = load_metadata(original_in_path)
        print("Loading original data ...")
        orig_samples, samples_meta = load_data_ceti_original(original_in_path, metadata)
    elif dataset_name == "Horst-Study":
        dataset = HorstFitDataset(original_in_path)
        metadata = dataset.subject_meta
        orig_samples = dict()
        samples_meta = dict()
        for ele in dataset.all_samples:
            orig_samples[ele[:-4]] = dataset.all_samples[ele]
            samples_meta[ele[:-4]] = dataset.sample_meta[ele]

    print("Loading fitted data...")
    fittings_samples = {}
    load_func = {"vposer": load_data_vposer_humor_fitted, "humor": load_data_humor_fitted}
    for fitting in fittings:
        fitted_in_path = osp.join(in_path, "fitted", "humor", dataset_name)
        fittings_samples[fitting] = load_func[fitting](fitted_in_path)

    #print(list(orig_samples.keys())[0])
    #print(list(fittings_samples["humor"].keys())[0])

    # Save subject metadata
    for subject in metadata:
        subject_out_path = osp.join(out_path, subject)
        metadata[subject]["participant_id"] = subject
        with open(subject_out_path + ".yaml", "w") as f:
            yaml.dump(dict(metadata[subject]), f)

    for sample in fittings_samples["humor"]:
        sample_file_name = sample.split("_")[0] + "." + "-".join(sample.split("_")[1:])
        sample_out_path = osp.join(out_path, sample_file_name)
        print(sample_file_name)

        orig_sample = orig_samples[sample]
        vposer_sample = fittings_samples["vposer"][sample]
        humor_sample = fittings_samples["humor"][sample]

        with (torch.no_grad()):

            vposer_sample["latent_pose"] = infer_vposer_latent_pose(vposer_sample)
            vposer_sample["latent_motion"] = infer_humor_latent_motion(vposer_sample)
            vposer_sample = produce_all_position_combinations(vposer_sample, dataset_name)
            vposer_sample["SMPL"] = np.concatenate((vposer_sample["pose_body"].flatten(), vposer_sample["trans"].flatten(), vposer_sample["root_orient"].flatten(), vposer_sample["betas"]))
            vposer_sample["SMPL_vposer"] = np.concatenate((vposer_sample["latent_pose"].flatten(), vposer_sample["trans"].flatten(), vposer_sample["root_orient"].flatten(), vposer_sample["betas"]))
            vposer_sample["SMPL_humor"] = np.concatenate((vposer_sample["latent_motion"].flatten(), vposer_sample["trans"].flatten(), vposer_sample["root_orient"].flatten(), vposer_sample["betas"]))

            humor_sample["latent_pose"] = infer_vposer_latent_pose(humor_sample)
            humor_sample["latent_motion"] = infer_humor_latent_motion(humor_sample)
            humor_sample = produce_all_position_combinations(humor_sample, dataset_name)
            humor_sample["SMPL"] = np.concatenate((humor_sample["pose_body"].flatten(), humor_sample["root_orient"].flatten(), humor_sample["trans"].flatten(), humor_sample["betas"]))
            humor_sample["SMPL_vposer"] = np.concatenate((humor_sample["latent_pose"].flatten(), humor_sample["root_orient"].flatten(), humor_sample["trans"].flatten(), humor_sample["betas"]))
            humor_sample["SMPL_humor"] = np.concatenate((humor_sample["latent_motion"].flatten(), humor_sample["root_orient"].flatten(), humor_sample["trans"].flatten(), humor_sample["betas"]))


        #print(orig_sample)
        #print(vposer_sample)
        prep_sample = create_prepared_data(orig_sample, vposer_sample, humor_sample)

        for ele in prep_sample:
            if torch.is_tensor(prep_sample[ele]):
                prep_sample[ele] = prep_sample[ele].detach().cpu()

        # Save data
        np.savez(sample_out_path, **prep_sample)

        # Save sample metadata
        with open(sample_out_path + ".yaml", "w") as f:
            if dataset_name == "Horst-Study":
                samples_meta[sample]["participant_id"] = sample.split("_")[0]
            yaml.dump(dict(samples_meta[sample]),f)


if __name__=='__main__':

    if len(sys.argv) < 2:
        print("Missing arguments")
        exit()

    in_path = sys.argv[1]
    dataset_name = sys.argv[2]
    fittings = ["humor", "vposer"]

    main(in_path, dataset_name, fittings)