import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import numpy as np
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import yaml

from anonymization.anon_viz import render_pose_seq

class AnonymizationDataset(Dataset):
    '''
    Loader for the CeTI Locomotion dataset.
    '''

    def __init__(self, data_path, meta_path=None,
                       split_size=None,
                       split_part=None):

        super(AnonymizationDataset, self).__init__()

        self.data_path = data_path

        self.meta_path = meta_path
        if self.meta_path is None:
            self.meta_path = data_path

        self.split_size = split_size
        self.split_part = split_part

        # All prepared data samples in the correct order
        self.all_samples = []

        # Maps the sequence number to the subject
        self.sequence_to_subject = {}

        # The metadata of the subjects
        self.subject_meta = {}

        # The metadata of the sequences
        self.sample_meta = {}

        self.load_all_data()



    def load_all_data(self):

        subject_metadata = self.load_subject_metadata(self.meta_path)

        # Only uses parts of the subjects
        if self.split_size:
            new_meta = dict()
            x = np.array(list(subject_metadata.keys()))
            meta_split = np.array_split(x, self.split_size)[self.split_part]

            for k in meta_split:
                new_meta[k] = subject_metadata[k]

            subject_metadata = new_meta

        samples, sample_meta = self.load_data(self.data_path, self.meta_path, subject_metadata)

        self.all_samples = samples
        self.subject_meta = subject_metadata
        self.sample_meta = sample_meta


    def load_subject_metadata(self, in_path):
        files = [f for f in listdir(in_path) if isfile(join(in_path, f))]
        subject_files = [f for f in files if len(f.split(".")) < 3]

        subjects = {}
        for file in subject_files:
            with open(join(in_path, file), "r") as f:
                data = yaml.safe_load(f)
                subjects[data["participant_id"]] = data

        return subjects

    def prepare_meta(self, data):
        for ele in ['age', 'arm_span', 'foot_length', 'height', 'hip_height', 'hip_width', 'knee_height',
                    'manus_length', 'shoulder_height', 'shoulder_width']:
            if ele in data:
                data[ele] = int(data[ele])

        for ele in ['mass', 'sts-1-time', 'sts-2-time']:
            if ele in data:
                data[ele] = float(data[ele])

        return data

    def load_data(self, in_path, in_path_meta, subjects_metadata):
        samples = {}
        samples_meta = {}

        files = [f for f in listdir(in_path) if isfile(join(in_path, f))]
        motion_files = [f for f in files if ".npz" in f]

        for file in motion_files:
            if file.split(".")[0] in subjects_metadata:
                with open(join(in_path, file), "rb") as f:
                    tmp = np.load(f)
                    samples[file] = {}
                    for ele in tmp:
                        samples[file][ele] = tmp[ele]

                with open(join(in_path_meta, file[:-3] + "yaml"), "r") as f:
                    samples_meta[file] = yaml.safe_load(f)
                    samples_meta[file] = self.prepare_meta(samples_meta[file])
                    samples_meta[file]["file_path"] = file

        return samples, samples_meta

    def __len__(self):
        return len(self.all_samples.keys())

    def __getitem__(self, idx):
        sample = list(self.all_samples.keys())[idx]
        data = self.all_samples[sample]
        meta = self.sample_meta[sample]

        return meta, data


if __name__=='__main__':

    #data_import_dir = join('../data/anon/CeTI-Locomotion/humor_fitting/vposer/normal_0.4')
    data_import_dir = join('../data/prepared/Horst-Study')
    data_meta_dir = join('../data/prepared/Horst-Study')

    dataset = AnonymizationDataset(data_import_dir, data_meta_dir)

    #print(len(dataset))
    #print(dataset.subject_meta)

    item = dataset.__getitem__(len(dataset) - 1)
    meta, data = item

    print(list(data.keys()))
    #foo = [data["original_positions"], data["vposer_positions"], data["humor_positions"]]
    #print(foo[0].shape)
    #print(foo[1].shape)

    for i, sample in enumerate(dataset):
        meta, data = sample

        for ele in data:
            if np.isnan(data[ele]).any():
                print(meta)
                print(ele)

    #render_pose_seq(foo, "test")
