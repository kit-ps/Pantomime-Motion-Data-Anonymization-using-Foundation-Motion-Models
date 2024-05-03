import json
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
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as rot

from body_model.utils import SMPLH_PATH, SMPL_JOINTS

from anonymization.anon_viz import render_pose_seq


HORST_TO_SMPL_JOINTS = {'hips' : ['L ILIAC', 'R ILIAC'],
                        'leftUpLeg' : ['L GTROCH'],
                        'rightUpLeg' : ['R GTROCH'],
                        'spine' : ['R BACK', 'L BACK', 'SACRUM'],
                        'leftLeg' : ['L THIGH 3', 'L THIGH 4', 'L SHANK 1', 'L SHANK 2'],
                        'rightLeg' : ['R THIGH 3', 'R THIGH 4', 'R SHANK 1', 'R SHANK 2'],
                        'spine1' : ['R BACK', 'L BACK'],
                        'leftFoot' : ['L HEEL'],
                        'rightFoot' : ['R HEEL'],
                        'spine2' : ['STERNUM'],
                        'leftToeBase' : ['L FOOT LAT', 'L FOOT MED'],
                        'rightToeBase' : ['R FOOT LAT', 'R FOOT MED'],
                        'neck' : ['C7'],
                        'leftShoulder' : ['L ACROMION', 'STERNUM'],
                        'rightShoulder' : ['R ACROMION', 'STERNUM'],
                        'head' : ['HEAD ANT', 'R HEAD', 'L HEAD'],
                        'leftArm' : ['L ACROMION'],
                        'rightArm' : ['R ACROMION'],
                        'leftForeArm' : ['L ELBOW LAT', 'L ELBOW MED'],
                        'rightForeArm' : ['R ELBOW LAT', 'R ELBOW MED'],
                        'leftHand' : ['L WRIST LAT', 'L WIRST MED'],
                        'rightHand' : ['R WIRST LAT', 'R WRITS MED']}

HORST_MARKERS_NOT_TO_USE = ["spine", "spine1", "spine2", "leftShoulder", "rightShoulder", "neck"]

HORST_MARKER_NAMES = ['HEAD ANT', 'R HEAD', 'L HEAD', 'C7', 'R BACK', 'L BACK', 'STERNUM', 'R ACROMION',
                      'R HUM PROX', 'R HUM ANT', 'R HUM POST', 'R ELBOW LAT', 'R ELBOW MED', 'R FOREARM', 'R WIRST LAT',
                      'R WRITS MED', 'R HAND', 'L ACROMION', 'L HUM PROX', 'L HUM ANT', 'L HUM POST', 'L ELBOW LAT',
                      'L ELBOW MED', 'L FOREARM', 'L WRIST LAT', 'L WIRST MED', 'L HAND', 'R ILIAC', 'SACRUM', 'L ILIAC',
                      'R GTROCH', 'R THIGH  1', 'R THIGH 2', 'R THIGH 3', 'R THIGH 4', 'R SHANK 1', 'R SHANK 2', 'R SHANK 3',
                      'R SHANK 4', 'R HEEL', 'R FOOT LAT', 'R FOOT MED', 'L GTROCH', 'L THIGH 1', 'L THIGH 2', 'L THIGH 3',
                      'L THIGH 4', 'L SHANK 1', 'L SHANK 2', 'L SHANK 3', 'L SHANK 4', 'L HEEL', 'L FOOT LAT', 'L FOOT MED']

HORST_BODY_PARTS = {
    "head": [0, 1, 2],
    "torso": [3, 4, 5, 6],
    "right_arm": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "left_arm": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    "hip": [27, 28, 29],
    "right_leg": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    "left_leg": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
}

HORST_REDUCED_BODY_PARTS = {
    "head": [0, 1, 2],
    "c7": [3],
    "torso": [4, 5, 6],
    "shoulder_R": [7],
    "elbow_R": [11, 12],
    "wrist_R": [14, 15],
    "hand_R": [16],
    "shoulder_L": [17],
    "elbow_L": [21, 22],
    "wrist_L": [24, 25],
    "hand_L:": [26],
    "hip": [27, 28, 29],  # 9
    "gtroch_R": [30],
    "thigh_R": [31, 32, 33, 34],
    "knee_R": [31, 32, 33, 34, 35, 36, 37, 38],
    "shank_R": [35, 36, 37, 38],
    "heel_R": [39],
    "foot_R": [40, 41],
    "gtroch_L": [42],
    "thigh_L": [43, 44, 45, 46],
    "knee_L": [43, 44, 45, 46, 47, 48, 49, 50],
    "shank_L": [47, 48, 49, 50],
    "heel_L": [51],
    "foot_L": [52, 53],
}

HORST_ANGLE_JOINTS = {
    "head_torso": [0, 1, 2],
    "shoulders": [3, 2, 6],
    "arm_R": [3, 4, 5],
    "arm_L": [6, 7, 8],
    "hip_knee_R": [9, 10, 11],
    "hip_knee_L": [9, 14, 15],
    "leg_R": [10, 11, 12],
    "leg_L": [14, 15, 16],
    "foot_R": [11, 12, 13],
    "foot_L": [15, 16, 17],
}

class HorstFitDataset(Dataset):
    '''
    Loader for the Horst Study dataset.
    '''

    def __init__(self, data_path,
                       return_joints=True,
                       return_points=True,
                       num_samp_pts=21,
                       full_sequence=True,
                       split_size=None,
                       split_part=None):

        super(HorstFitDataset, self).__init__()

        self.return_joints = return_joints
        self.return_points = return_points
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


    def load_force_plates(self, id, point, input_path):
        start = None
        with open(join(input_path, "Gait_rawdata_f_1_tsv/", id + "_" + point + "_Gait_f_1.tsv")) as f:
            data = list(csv.reader(f, delimiter="\t"))
            for i in range(24, len(data)):
                if float(data[i][4]) > 10:
                    start = float(data[i][1])
                    break

        end = None
        with open(join(input_path, "Gait_rawdata_f_2_tsv/", id + "_" + point + "_Gait_f_2.tsv")) as f:
            data = list(csv.reader(f, delimiter="\t"))

            flag = False
            for i in range(24, len(data)):
                if float(data[i][4]) > 20:
                    flag = True

                if float(data[i][4]) < 20 and flag:
                    end = float(data[i][1])
                    break

        return (start, end)

    def reduce_markers(self, result):
        new_result = []
        for pose in result:
            new_pose = []
            for marker in HORST_REDUCED_BODY_PARTS:
                seq = HORST_REDUCED_BODY_PARTS[marker]
                new_x = 0
                new_y = 0
                new_z = 0
                for i in seq:
                    new_x += pose[3 * i]
                    new_y += pose[3 * i + 1]
                    new_z += pose[3 * i + 2]
                new_x /= len(seq)
                new_y /= len(seq)
                new_z /= len(seq)
                new_pose.append(new_x)
                new_pose.append(new_y)
                new_pose.append(new_z)

            new_result.append(new_pose)
        return new_result

    def positions_to_angles(self, data):
        new_result = []
        for pose in data:
            new_pose = []
            for angle in HORST_ANGLE_JOINTS:
                seq = np.array(HORST_ANGLE_JOINTS[angle])

                vec1 = [
                    pose[seq[0] * 3] - pose[seq[1] * 3],
                    pose[seq[0] * 3 + 1] - pose[seq[1] * 3 + 1],
                    pose[seq[0] * 3 + 2] - pose[seq[1] * 3 + 2],
                ]
                vec2 = [
                    pose[seq[2] * 3] - pose[seq[1] * 3],
                    pose[seq[2] * 3 + 1] - pose[seq[1] * 3 + 1],
                    pose[seq[2] * 3 + 2] - pose[seq[1] * 3 + 2],
                ]
                tmp1 = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
                tmp2 = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                angle_val = np.arccos(tmp1 / tmp2)
                new_pose.append(angle_val)
            new_result.append(new_pose)

        return list(new_result)


    def horst_to_smpl_joints(self, data):

        for ele in HORST_TO_SMPL_JOINTS:
            horst_marker = HORST_TO_SMPL_JOINTS[ele]




    def load_data(self, input_path, metadata):
        in_path = join(input_path, "Gait_rawdata_tsv")

        samples = {}
        samples_meta = {}
        files = [f for f in listdir(in_path) if isfile(join(in_path, f))]
        motion_files = [f for f in files if "_Gait.tsv" in f]

        for file in motion_files:

            parts = file[:-4].split("_")
            subject_id = parts[0]
            point_id = parts[1]

            if subject_id in metadata:
                with open(join(in_path, file)) as tsvFile:
                    data = list(csv.reader(tsvFile, delimiter="\t"))
                    result = []
                    time = []
                    for i in range(11, len(data)):
                        result.append([float(i) for i in data[i][2:]])
                        time.append([float(i) for i in data[i][1:2]])

                    plates = self.load_force_plates(subject_id, point_id, input_path)

                    start = 0
                    end = 0
                    for i in range(len(time)):
                        frame = time[i][0]
                        if start == 0 and frame > plates[0]:
                            start = i

                        if frame > plates[1]:
                            end = i
                            break

                    samples[file] = np.array(result[start:end])
                    samples[file] = self.prepare_sequence(samples[file])
                    samples_meta[file] = copy.deepcopy(metadata[subject_id])
                    samples_meta[file]["file_path"] = file

        return samples, samples_meta


    def load_metadata(self, input_path):
        with open(join(input_path, "gait_subject_info.csv")) as f:
            metadata = list(csv.reader(f, delimiter=";"))
        metadata_mat = loadmat(join(input_path, "Gait_GRF_JA_Label.mat"))
        gender_per_id = {}
        for i in range(len(metadata_mat["Feature_JA_Full"])):
            sex = metadata_mat["Target_Gender"][i][0]
            subject_id = str(np.where(metadata_mat["Target_SubjectID"][i] == 1)[0][0] + 1)
            if subject_id not in gender_per_id:
                gender_per_id[subject_id] = sex
        metadata_output = {}
        for line in metadata[1:]:
            metadata_result = {}
            sex = {"0": "male", "1": "female"}
            subject_id = str(line[0][1:])
            metadata_result["id"] = subject_id
            metadata_result["sex"] = sex[str(gender_per_id[subject_id])]
            metadata_result["age"] = int(line[2])
            metadata_result["mass"] = float(line[3].replace(",", "."))
            metadata_result["height"] = float(line[4].replace(",", "."))
            # Fix the name difference between the subject ids and the file subject ids
            if len(subject_id) == 1 and subject_id != "1":
                subject_id = '0' + str(subject_id)
            metadata_output['S' + str(subject_id)] = metadata_result

        return metadata_output

    def prepare_sequence(self, sequence):
        new_data = []
        for i in range(len(sequence[0])):
            dim = sequence[:, i]

            old_times = np.arange(len(dim))
            # The original fps of CeTI Locomotion is 250 Hz, downsampling to 30 Hz
            step = 250 / 30
            new_times = np.arange(old_times[0], old_times[-1], step)

            new_dim = np.interp(new_times, old_times, dim)
            new_data.append(new_dim)

        # Transpose as now as list of the dimensions, not of the frames
        new_data = np.array(new_data).T

        # Device by 1000 to have the right unit
        new_data = new_data.reshape(-1, 54, 3) / 1000

        # Rotate to make walker walk in direction of the z-axis
        rotation_matrix = rot.from_rotvec(np.array([0, 0, 90]), degrees=True).as_matrix()
        new_data = np.dot(new_data, rotation_matrix)
        rotation_matrix = rot.from_rotvec(np.array([90, 0, 0]), degrees=True).as_matrix()
        new_data = np.dot(new_data, rotation_matrix)


        new_data = new_data


        return torch.tensor(np.array(new_data)).type(torch.float)


    # Returns the subset of SMPL markers calculated from the Horst Markers.
    @staticmethod
    def original_data_to_smpl_full(sample):
        out_sample = np.ones((len(sample), len(SMPL_JOINTS), 3)) * float(1000)

        for k, v in SMPL_JOINTS.items():
            #if k not in HORST_MARKERS_NOT_TO_USE:
                marker_index = []
                for ele in HORST_TO_SMPL_JOINTS[k]:
                    marker_index.append(HORST_MARKER_NAMES.index(ele))
                if len(marker_index) == 1:
                    out_sample[:, v, :] = sample[:, marker_index, :].reshape(-1, 3)
                else:
                    out_sample[:, v, :] = sample[:, marker_index, :].mean(axis=1)

        return out_sample


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
                if k not in HORST_MARKERS_NOT_TO_USE:
                    marker_index = []
                    for ele in HORST_TO_SMPL_JOINTS[k]:
                        marker_index.append(HORST_MARKER_NAMES.index(ele))
                    if len(marker_index) == 1:
                        observed_dict["joints3d"][:,v,:] = global_data[:, marker_index, :].reshape(-1, 3)
                    else:
                        observed_dict["joints3d"][:, v, :] = global_data[:, marker_index, :].mean(axis=1)

        if self.return_points:
            #points = torch.Tensor(np.stack(points_list, axis=0))
            observed_dict['points3d'] = global_data


        gt_dict = copy.deepcopy(observed_dict)
        gt_dict["name"] = join(self.data_path, sample)
        return observed_dict, gt_dict


def profile_test():
    dataset = HorstFitDataset(data_import_dir, return_joints=True, return_points=True, num_samp_pts=512)
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)


if __name__=='__main__':
    # dataset
    data_rot_rep = 'mat'

    data_import_dir = join('../data/original/Horst_Study/')

    dataset = HorstFitDataset(data_import_dir, return_joints=True, return_points=True, num_samp_pts=512)

