'''
Custom dataset for brain tumor classification, single modal input
Sidi Liang, 2022-2023
'''

import math
import os
import pathlib
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
from typing import Tuple, List, Dict

class CustomTumorDataset(Dataset):

    def __init__(self, root_dir, sets):
        self.root_dir = root_dir
        self.paths = list(pathlib.Path(root_dir).glob("*/*.nii.gz")) + list(pathlib.Path(root_dir).glob("*/*.nii"))
        self.classes, self.class_to_idx = self.__find_classes__(root_dir)
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")

        return new_data

    # Make function to find classes in target directory
    def __find_classes__(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folder names in a target directory.

        Assumes target directory is in standard image classification format.

        Args:
            directory (str): target directory to load classnames from.

        Returns:
            Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

        Example:
            find_classes("food_images/train")
            >>> (["class_1", "class_2"], {"class_1": 0, ...})
        """
        # 1. Get the class names by scanning the target directory
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        # 2. Raise an error if class names not found
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

        # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        if self.phase == "train":
            # read image and labels
            img_path = self.paths[idx]
            class_name  = self.paths[idx].parent.name
            assert os.path.isfile(img_path)
            img = nibabel.load(img_path)  # We have transposed the data from WHD format to DHW
            assert img is not None

            # data processing
            img_array = self.__training_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            class_idx = self.class_to_idx[class_name]

            return img_array, class_idx, img_path.__str__()

        elif self.phase == "test":
            #WIP
            # read image
            img_path = self.paths[idx]
            class_name  = self.paths[idx].parent.name
            print(img_path)
            assert os.path.isfile(img_path)
            img = nibabel.load(img_path)
            assert img is not None

            # data processing
            img_array = self.__testing_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)

            return img_array


    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]


    def __random_center_crop__(self, data, label=None):
        from random import random
        """
        Random crop
        """
        if label is not None:
            target_indexs = np.where(label>0)
        else:
            target_indexs = np.where(data>0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth*1.0/2) * random())
        Y_min = int((min_H - target_height*1.0/2) * random())
        X_min = int((min_W - target_width*1.0/2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth*1.0/2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height*1.0/2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width*1.0/2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        if label is None:
            return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]
        else:
            return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]



    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [width, depth, height] = data.shape
        scale = [self.input_W*1.0/width, self.input_D*1.0/depth, self.input_H*1.0/height]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __crop_data__(self, data, label=None):
        """
        Random crop with different methods:
        """
        if label is None:
            # random center crop
            data = self.__random_center_crop__ (data)
            return data
        # random center crop
        data, label = self.__random_center_crop__ (data, label)
        return data, label

    def __training_data_process__(self, data, label=None):
        if label is None:
            # For classification
            # crop data according net input size
            data = data.get_data()
           # data = data[:,:,:,0]
            # drop out the invalid range
            data = self.__drop_invalid_range__(data)

            # crop data
            #data = self.__crop_data__(data)

            # resize data
            data = self.__resize_data__(data)

            # normalization datas
            data = self.__itensity_normalize_one_volume__(data)

            return data
        else:
            # crop data according net input size
            # For segmentation, WIP
            data = data.get_data()
            label = label.get_data()

            # drop out the invalid range
            data, label = self.__drop_invalid_range__(data, label)

            # crop data
            data, label = self.__crop_data__(data, label)

            # resize data
            data = self.__resize_data__(data)
            label = self.__resize_data__(label)

            # normalization datas
            data = self.__itensity_normalize_one_volume__(data)
            return data, label

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()
       # data = data[:,:,:,0]

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data
