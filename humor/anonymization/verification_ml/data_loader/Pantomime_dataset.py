import os

from torch.utils.data import Dataset
import torch

import numpy as np
from os.path import join

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tqdm import tqdm



class PantomimeDataset(Dataset):

    # Resamples the give motion sequence to 100 frames and in the end flattens the sequence into one vector
    @staticmethod
    def resample(data, num_frames=100):
        new_data = []
        for i in range(len(data[0])):
            dim = data[:, i]

            old_times = np.arange(len(dim))
            step = (old_times[-1] - old_times[0]) / (num_frames)
            new_times = np.arange(old_times[0], old_times[-1], step)

            # Sometimes the rounding gives you an extra step, therefore this makes sure we only have num_frames in total
            new_times = new_times[0:num_frames]

            new_dim = np.interp(new_times, old_times, dim)
            new_data.append(new_dim)

        return np.nan_to_num(np.array(new_data))



    def __init__(self, path_str, train=False):

        self.path = join(path_str)

        self.train = train

        self.load_data(self.path)

        #if "Horst" in path_str:
        #    self.frame_size = 66
        #else:
        #    self.frame_size = 51

        self.frame_size = self.data[0].shape[0]


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels,
                                                                                   test_size=0.2,
                                                                                   stratify=self.labels)

        if self.train:
            self.labels = self.y_train
            self.data = self.X_train
        else:
            self.labels = self.y_test
            self.data = self.X_test




    def load_data(self, path):

        all_files = os.listdir(path)

        sample_files = [f for f in all_files if ".npz" in f]

        self.labels = []
        self.data = []
        for file in tqdm(sample_files):
            label = file.split(".")[0]
            data = np.load(join(path, file))["positions"]

            #print(data.shape)
            data = PantomimeDataset.resample(data, num_frames=100)

            #data = data.flatten()
            #print(data.shape)
            data = torch.from_numpy(data)
            data = data.type(torch.float32)

            self.data.append(data)
            self.labels.append(label)

        """
        labels_count = {}
        for label in self.labels:
            if label not in labels_count:
                labels_count[label] = 1
            else:
                labels_count[label] += 1

        max = 0
        tmp_l = None
        for label in labels_count:
            if labels_count[label] > max:
                max = labels_count[label]
                tmp_l = label
        print("Max class: ", tmp_l, max, max / len(self.labels))
        """

        # Numeric encoding of the labels
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.labels = le.transform(self.labels)
        self.labels = torch.as_tensor(self.labels)




    def get_num_classes(self):
        label_list = []
        for label in self.labels:
            label = label.item()
            if label not in label_list:
                label_list.append(label)

        return len(label_list)


    def switch_to_test(self):
        self.labels = self.y_test
        self.data = self.X_test


    def switch_to_train(self):
        self.labels = self.y_train
        self.data = self.X_train


    def get_frame_size(self):
        return self.frame_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    dataset = PantomimeDataset("/home/simon/DuD/research/pantomime/code/data/original/HuMMan")

    '''
    dataset = FacialMotionDataset("../processing/output/data/derivatives/", get_triplet=False, train=False, flatten=False,
                                            resample=True, add_padding=False, data_type="hmd", filter_pilot=True,
                                            label_attribute="Tasks_Name", scale_data=False, headsets=["vive", "meta", "pico"],
                                            split_type="all", task_categories=["Animation"], combine_data_types=False,
                                            text_split_type="tasks")
    '''