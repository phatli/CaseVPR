import json
import os

import cv2
from ..utils import Mat_Redis_Utils



class NordlandV():
    def __init__(self, name, DATASET_DIR, use_redis=True):
        """NordlandV class

        Args:
            name (str): name of the dataset, can be "spring", "summer", "fall" or "winter"
            DATASET_DIR (str): path to the dataset directory
        """
        self.name = name
        self.path = os.path.join(DATASET_DIR, f"test_{name}")
        self.gt_path = os.path.join(DATASET_DIR, f"gt_test_{name}.json")
        with open(self.gt_path, 'r') as f:
            self.gt = json.load(f)
        self.len = len(os.listdir(self.path))
        if use_redis:
            self.redis = Mat_Redis_Utils()
        self.use_redis = use_redis

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.use_redis:
            img = self.redis.load_cv2(os.path.join(self.path, f"{idx}.png"))
        else:
            img = cv2.imread(os.path.join(self.path, f"{idx}.png"))
        pose = [self.gt[idx], 0]
        return img, pose


class Robotcar():
    def __init__(self, name, DATASET_DIR, use_redis=True):
        """Initialize Robotcar dataset.

        Args:
            name (str): name of the dataset, i.e. 'test_queries' or 'train_database'.
            DATASET_DIR (str): path to the dataset directory.
        """
        self.split_name = name.split('_')[0]
        self.name = name.split('_')[1]
        self.path = os.path.join(
            DATASET_DIR, f"{self.split_name}", f"{self.name}", "sequence")
        self.len = len(os.listdir(self.path))
        self.images = {}
        for image_file in os.listdir(self.path):
            image_data = image_file.split("@")
            self.images[int(image_data[4])] = {
                "northing": float(image_data[1]),
                "easting": float(image_data[2]),
                "date": image_data[3],
                "timestamp": int(image_data[5]),
                "file": os.path.join(self.path, image_file)
            }
        if use_redis:
            self.redis = Mat_Redis_Utils()
        self.use_redis = use_redis

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx in self.images.keys(), "Index out of range"
        image = self.images[idx]
        image_path = image["file"]
        if self.use_redis:
            img = self.redis.load_cv2(image_path)
        else:
            img = cv2.imread(image_path)
        return img, (image["northing"], image["easting"])


class RosImg():
    def __init__(self, name, DATASET_DIR, use_redis=True):
        """Initialize ROS dataset.

        Args:
            name (str): name of the dataset, i.e. 'nanyanglink_ccw_day_2_210622' or 'src_ccw_day_120922'.
            DATASET_DIR (str): path to the dataset directory.
        """
        self.path = os.path.join(DATASET_DIR, f"{name}")
        self.gt_path = os.path.join(DATASET_DIR, f"{name}.json")
        with open(self.gt_path, 'r') as f:
            self.gt = json.load(f)
        self.len = len(os.listdir(self.path))
        if use_redis:
            self.redis = Mat_Redis_Utils()
        self.use_redis = use_redis

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.use_redis:
            img = self.redis.load_cv2(os.path.join(self.path, f"{idx + 1}.png"))
        else:
            img = cv2.imread(os.path.join(self.path, f"{idx + 1}.png"))
        pose = self.gt[idx]["pos"]
        return img, pose
