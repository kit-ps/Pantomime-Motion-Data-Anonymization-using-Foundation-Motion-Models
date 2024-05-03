import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import numpy as np
from torch.utils.data import Dataset
import torch
from os import listdir
from os.path import isfile, join
import copy
import csv

from body_model.utils import SMPLH_PATH, SMPL_JOINTS

from anonymization.anon_viz import render_pose_seq

# Matches the names of the SMPL joints to the position of the corresponding ceti joint in the data
CETI_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 4, 'spine' : None, 'leftLeg' : 2, 'rightLeg' : 5,
                'spine1' : None, 'leftFoot' : 3, 'rightFoot' : 6, 'spine2' : 7, 'leftToeBase' : None, 'rightToeBase' : None,
                'neck' : None, 'leftShoulder' : 8, 'rightShoulder' : 13, 'head' : 12, 'leftArm' : 9, 'rightArm' : 14,
                'leftForeArm' : 10, 'rightForeArm' : 15, 'leftHand' : 11, 'rightHand' : 16}

CETI_JOINTS_reduced = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 4, 'spine' : None, 'leftLeg' : 2, 'rightLeg' : 5,
                'spine1' : None, 'leftFoot' : 3, 'rightFoot' : 6, 'spine2' : None, 'leftToeBase' : None, 'rightToeBase' : None,
                'neck' : None, 'leftShoulder' : None, 'rightShoulder' : None, 'head' : 12, 'leftArm' : 9, 'rightArm' : 14,
                'leftForeArm' : 10, 'rightForeArm' : 15, 'leftHand' : 11, 'rightHand' : 16}


# The SMPL JOINT Names in the correct order
SMPL_JOINTS_NAMES = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg', 'spine1', 'leftFoot',
                     'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase', 'neck', 'leftShoulder', 'rightShoulder',
                     'head', 'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand']


CETI_MARKER_NAMES = ['Pelvis_position_x', 'Pelvis_position_y', 'Pelvis_position_z',
                     'LeftUpperLeg_position_x', 'LeftUpperLeg_position_y', 'LeftUpperLeg_position_z',
                     'LeftLowerLeg_position_x', 'LeftLowerLeg_position_y', 'LeftLowerLeg_position_z',
                     'LeftFoot_position_x', 'LeftFoot_position_y', 'LeftFoot_position_z',
                     'RightUpperLeg_position_x', 'RightUpperLeg_position_y', 'RightUpperLeg_position_z',
                     'RightLowerLeg_position_x', 'RightLowerLeg_position_y', 'RightLowerLeg_position_z',
                     'RightFoot_position_x', 'RightFoot_position_y', 'RightFoot_position_z',
                     'Chest_position_x', 'Chest_position_y', 'Chest_position_z',
                     'LeftShoulder_position_x', 'LeftShoulder_position_y', 'LeftShoulder_position_z',
                     'LeftUpperArm_position_x', 'LeftUpperArm_position_y', 'LeftUpperArm_position_z',
                     'LeftForeArm_position_x', 'LeftForeArm_position_y', 'LeftForeArm_position_z',
                     'LeftHand_position_x', 'LeftHand_position_y', 'LeftHand_position_z',
                     'Head_position_x', 'Head_position_y', 'Head_position_z',
                     'RightShoulder_position_x', 'RightShoulder_position_y', 'RightShoulder_position_z',
                     'RightUpperArm_position_x', 'RightUpperArm_position_y', 'RightUpperArm_position_z',
                     'RightForeArm_position_x', 'RightForeArm_position_y', 'RightForeArm_position_z',
                     'RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z',
                     'Pelvis_extension', 'Pelvis_lateral_flexion_rotation', 'Pelvis_axial_rotation',
                     'LeftHip_flexion', 'LeftHip_adduction', 'LeftHip_external_rotation',
                     'LeftKnee_flexion', 'LeftKnee_adduction', 'LeftKnee_external_rotation',
                     'LeftAnkle_dorsiflexion', 'LeftAnkle_inversion', 'LeftAnkle_internal_rotation',
                     'RightHip_flexion', 'RightHip_adduction', 'RightHip_external_rotation',
                     'RightKnee_flexion', 'RightKnee_adduction', 'RightKnee_external_rotation',
                     'RightAnkle_dorsiflexion', 'RightAnkle_inversion', 'RightAnkle_internal_rotation',
                     'Thorax_extension', 'Thorax_lateral_flexion_rotation', 'Thorax_axial_rotation',
                     'LeftScapula_protraction', 'LeftScapula_medial_rotation', 'LeftScapula_posterior_tilt',
                     'LeftShoulder_flexion', 'LeftShoulder_abduction', 'LeftShoulder_external_rotation',
                     'LeftElbow_flexion', 'LeftElbow_abduction', 'LeftElbow_pronation',
                     'LeftWrist_flexion', 'LeftWrist_abduction', 'LeftWrist_pronation',
                     'Neck_flexion', 'Neck_left-ward_tilt', 'Neck_right-ward_rotation',
                     'RightScapula_protraction', 'RightScapula_medial_rotation', 'RightScapula_posterior_tilt',
                     'RightShoulder_flexion', 'RightShoulder_abduction', 'RightShoulder_external_rotation',
                     'RightElbow_flexion', 'RightElbow_abduction', 'RightElbow_pronation',
                     'RightWrist_flexion', 'RightWrist_abduction', 'RightWrist_pronation']


class CeTIFitDataset(Dataset):
    '''
    Loader for the CeTI Locomotion dataset.
    '''

    def __init__(self, data_path,
                       seq_len=60,
                       return_joints=True,
                       return_points=True,
                       return_rotations=False,
                       num_samp_pts=17,
                       full_sequence=True,
                       split_size=None,
                       split_part=None):

        super(CeTIFitDataset, self).__init__()

        self.seq_len = seq_len # global seq returns + 1
        self.return_joints = return_joints
        self.return_points = return_points
        self.return_rotations = return_rotations
        self.num_samp_pts = num_samp_pts
        self.data_path = data_path
        self.full_sequence = full_sequence

        self.split_size = split_size
        self.split_part = split_part

        # All prepared data samples in the correct order
        self.all_samples = []

        # Maps IDX of a sample to the sequence it belongs to
        self.sub_sequence_map = {}

        # Maps the sequence number to the subject
        self.sequence_to_subject = {}

        # The metadata of the subjects
        self.subject_meta = {}

        # The metadata of the sequences
        self.sequence_meta = {}

        self.load_all_data()


    def load_all_data(self):

        metadata = self.load_metadata(self.data_path)

        # Only uses parts of the subjects
        if self.split_size:
            new_meta = dict()
            x = np.array(list(metadata.keys()))
            meta_split = np.array_split(x, self.split_size)[self.split_part]

            for k in meta_split:
                new_meta[k] = metadata[k]

            metadata = new_meta

        samples, sample_meta = self.load_data(self.data_path, metadata)

        self.all_samples = samples
        self.subject_meta = metadata
        self.sample_meta = sample_meta

        self.sub_sequence_map = dict()

        k = 0
        for sample in self.all_samples:
            cur_sample = self.all_samples[sample]
            for i in range(len(cur_sample) - 1):
                self.sub_sequence_map[k] = (sample, i)
                k += 1

    def load_metadata(self, in_path):
        with open(join(in_path, 'participants.tsv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
            metadata = {}
            for row in reader:
                metadata[row["participant_id"]] = row

            return metadata

    def load_data(self, in_path, metadata):
        samples = {}
        samples_meta = {}
        for participant in metadata:

            participant_path = join(in_path, participant, "motion")
            files = [f for f in listdir(participant_path) if isfile(join(participant_path, f))]
            motion_files = [f for f in files if "_motion.tsv" in f]

            for file in motion_files:
                with open(join(participant_path, file)) as tsv_file:
                    reader = csv.reader(tsv_file, delimiter='\t', quotechar='"')
                    samples[file] = np.array(list(reader)[1:], dtype=np.float32)
                    # Some preprocessing for the classification
                    samples[file] = self.prepare_sequence(samples[file])
                    samples_meta[file] = copy.deepcopy(metadata[participant])
                    samples_meta[file]["modality"] = file.split("_")[1]
                    samples_meta[file]["file_path"] = file

        return samples, samples_meta



    def prepare_sequence(self, sequence):
        # Only select the 17 joint positions, not the joint angles
        sequence = sequence[:,:17 * 3]

        new_data = []
        for i in range(len(sequence[0])):
            dim = sequence[:, i]

            old_times = np.arange(len(dim))
            # The original fps of CeTI Locomotion is 100 Hz, downsampling to 30 Hz
            step = 100 / 30
            new_times = np.arange(old_times[0], old_times[-1], step)

            new_dim = np.interp(new_times, old_times, dim)
            new_data.append(new_dim)

        # Transpose as now as list of the dimensions, not of the frames
        new_data = np.array(new_data).T.reshape(-1, 17, 3)

        return torch.tensor(np.array(new_data)).type(torch.float)

    @staticmethod
    def SMPL_joint_selection_mask():
        index_selected_smpl = []
        for joint in SMPL_JOINTS_NAMES:
            if CETI_JOINTS[joint] is not None:
                index_selected_smpl.append(SMPL_JOINTS[joint])

        index_selected_smpl = torch.tensor(index_selected_smpl)

        return index_selected_smpl

    @staticmethod
    def reduce_SMPL_joint_pos_to_reduced_ceti_joint_pos(positions):
        pos_out = np.ones((len(positions), len(SMPL_JOINTS), 3)) * float(10000)

        for k, v in SMPL_JOINTS.items():
            if CETI_JOINTS_reduced[k] is not None:
                pos_out[:, v, :] = positions[:, v, :]

        return pos_out

    @staticmethod
    def reduce_CeTI_joint_pos_to_reduced_ceti_joint_pos(positions):
        pos_out = np.ones((len(positions), len(SMPL_JOINTS), 3)) * float(10000)

        JOINT_ORDER = ['hips', 'leftUpLeg', 'rightUpLeg', 'leftLeg', 'rightLeg', 'leftFoot', 'rightFoot', 'spine2',
                       'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm',
                       'leftHand', 'rightHand']

        i = 0
        for ele in JOINT_ORDER:
            if CETI_JOINTS_reduced[ele] is not None:
                v = CETI_JOINTS_reduced[ele]
                pos_out[:, v, :] = positions[:, i, :]
            i += 1

        return pos_out


    @staticmethod
    def reduce_CeTI_original_joint_pos_to_reduced_ceti_joint_pos(positions):
        pos_out = np.ones((len(positions), len(SMPL_JOINTS), 3)) * float(10000)

        for k, v in SMPL_JOINTS.items():
            if CETI_JOINTS_reduced[k] is not None:
                pos_out[:, v, :] = positions[:, CETI_JOINTS_reduced[k], :]

        return pos_out


    def __len__(self):
        if self.full_sequence:
            return len(self.all_samples.keys())
        else:
            return len(self.sub_sequence_map.keys())

    def __getitem__(self, idx):
        if self.full_sequence:
            sample = list(self.all_samples.keys())[idx]
            global_data = self.all_samples[sample]
        else:
            sample, i = self.sub_sequence_map[idx]
            global_data = self.all_samples[sample][i:i+2]

        # create the ground truth data dictionary
        gt_dict = dict()

        # create clean observations
        observed_dict = dict()

        observed_dict["meta"] = self.sample_meta[sample]

        if self.return_joints:
            # 3d joint positions

            # Initialize all joints as unknown
            observed_dict["joints3d"] = np.ones((len(global_data), len(SMPL_JOINTS), 3)) * float('inf')

            for k, v in SMPL_JOINTS.items():
                if CETI_JOINTS[k] is not None:
                    observed_dict["joints3d"][:,v,:] = global_data[:, CETI_JOINTS[k], :]

        if self.return_rotations:
            # Initialize all joints as unknown, -1 because there is one joint less
            observed_dict["pose_body"] = np.ones((len(global_data), len(SMPL_JOINTS), 3)) * float('inf')

            for k, v in SMPL_JOINTS.items():
                # Root needs to be ignored, SMPL only has 21 joints
                if CETI_JOINTS[k] is not None:
                    observed_dict["pose_body"][:,v,:] = global_data[:, CETI_JOINTS[k], :]

            observed_dict["pose_body"] = observed_dict["pose_body"][:,1:]


        if self.return_points:
            #points = torch.Tensor(np.stack(points_list, axis=0))
            observed_dict['points3d'] = global_data


        gt_dict = copy.deepcopy(observed_dict)
        gt_dict["name"] = join(self.data_path, sample)
        return observed_dict, gt_dict


if __name__=='__main__':
    data_import_dir = join('../data/original/CeTI-Locomotion/')

    dataset = CeTIFitDataset(data_import_dir, seq_len=60, return_joints=True, return_points=True, num_samp_pts=512)

    print(dataset.__getitem__(10)[0]['points3d'].shape)
    print(dataset.__getitem__(10)[0]['joints3d'])
    print(dataset.__getitem__(10)[0]['joints3d'].shape)

    item = dataset.__getitem__(10)[0]['joints3d'][0]

    for i in range(item.shape[0]):
        if float("inf") in item[i]:
            print("Missing")

    render_pose_seq([dataset.__getitem__(10)[0]['joints3d'].reshape(-1, 22 * 3), dataset.__getitem__(10)[0]['joints3d'].reshape(-1, 22 * 3)], "test_fit_dataset")
