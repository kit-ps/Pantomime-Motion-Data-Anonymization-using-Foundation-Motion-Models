from torchvision import datasets, transforms
from base import BaseDataLoader

from .Pantomime_dataset import PantomimeDataset


"""
class FacialMotionDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, get_triplet=False, flatten=True, resample=False,
                 headsets=["vive", "meta", "pico"], data_type="hmd", split_type="all", scale_data=False, add_padding=False, shuffle=True,
                 validation_split=0.0, num_workers=1, training=True, label_attribute="participant_id",
                 task_categories=["Text", "Animation"], combine_data_types=False, text_split_type="tasks"):
        self.data_dir = data_dir
        self.dataset = FacialMotionDataset(self.data_dir, headsets=headsets, data_type=data_type, split_type=split_type,
                                           get_triplet=get_triplet, flatten=flatten, resample=resample, scale_data=scale_data, add_padding=add_padding, train=training,
                                           label_attribute=label_attribute, task_categories=task_categories, combine_data_types=combine_data_types, text_split_type=text_split_type)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

"""


class PantomimeDataLoader(BaseDataLoader):
    """
    HuMMan Data Loader
    """
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
        self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle=shuffle, validation_split=validation_split, num_workers=num_workers)
