import shutil
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))


from os import path as osp
from os import listdir
from pathlib import Path

from fitting.config import parse_args

from datasets.anonymization_dataset import AnonymizationDataset
from datasets.ceti_fit_dataset import CeTIFitDataset
from datasets.horst_fit_dataset import HorstFitDataset


def get_complete_stage3_fits(in_path):
    samples = []
    folders = [f for f in listdir(in_path) if not osp.isfile(osp.join(in_path, f))]
    motion_folders = [f for f in folders if "tsv" in f]

    for folder in motion_folders:
        file_path = osp.join(in_path, folder, "stage3_results.npz")

        if osp.isfile(file_path):
            samples.append(folder)

    return samples


def create_folder_with_missing_sequences(input_data_path, fit_data_path, dataset_type):
    if dataset_type == 'CeTI':
        cmpl_fit_path = osp.join(fit_data_path, "data/original/CeTI-Locomotion/derivatives/cut_sequences")
        completed_fits = get_complete_stage3_fits(cmpl_fit_path)
        fit_dataset = CeTIFitDataset(input_data_path, return_joints=True)
    else:
        completed_fits = get_complete_stage3_fits(osp.join(fit_data_path, "data/original/Horst-Study"))
        fit_dataset = HorstFitDataset(input_data_path, return_joints=True)

    original_keys, completed_fits_keys = [], []
    for fit in completed_fits:
        key = fit.split(".")[0]
        key = "".join(key.split("-"))
        key = "".join(key.split("_"))
        completed_fits_keys.append(key)

    for i, data in enumerate(fit_dataset):
        observed_dict, gt_dict = data
        file_path = observed_dict["meta"]["file_path"]
        key = file_path.split(".")[0]
        key = "".join(key.split("_"))
        key = "".join(key.split("-"))
        original_keys.append((key, file_path))

    print("Number of keys in original:", len(original_keys))
    print("Number of keys in fitted:", len(completed_fits_keys))

    missing_sequences = []
    for key, file_path in original_keys:
        if key not in completed_fits_keys:
            missing_sequences.append(file_path)

    if len(missing_sequences) > 0:
        # Removes the trailing backslash and creates a new folder name
        folder_path = input_data_path[:-1] + "_missing_sequences"

        shutil.rmtree(folder_path, ignore_errors=True)
        os.makedirs(folder_path)

        if dataset_type == 'CeTI':
            shutil.copyfile(osp.join(input_data_path, "participants.tsv"), osp.join(folder_path, "participants.tsv"))
        else:
            os.makedirs(osp.join(folder_path, "Gait_rawdata_tsv"), exist_ok=True)
            shutil.copytree(osp.join(input_data_path, "Gait_rawdata_f_1_tsv"), osp.join(folder_path, "Gait_rawdata_f_1_tsv"), True)
            shutil.copytree(osp.join(input_data_path, "Gait_rawdata_f_2_tsv"), osp.join(folder_path, "Gait_rawdata_f_2_tsv"), True)
            shutil.copyfile(osp.join(input_data_path, "gait_subject_info.csv"), osp.join(folder_path, "gait_subject_info.csv"))
            shutil.copyfile(osp.join(input_data_path, "Gait_GRF_JA_Label.mat"), osp.join(folder_path, "Gait_GRF_JA_Label.mat"))

        for file in missing_sequences:
            if dataset_type == "CeTI":
                src_file = osp.join(input_data_path, file.split("_")[0], "motion", file)
                dst_folder = osp.join(folder_path, file.split("_")[0], "motion")
                os.makedirs(dst_folder, exist_ok=True)
                shutil.copyfile(src_file, osp.join(dst_folder, file))
            else:
                src_file = osp.join(input_data_path, "Gait_rawdata_tsv", file)
                dst_file = osp.join(folder_path, "Gait_rawdata_tsv", file)
                shutil.copyfile(src_file, dst_file)



if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]

    create_folder_with_missing_sequences(args.data_path, args.out, args.data_type)