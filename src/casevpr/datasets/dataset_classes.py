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
            name (str): dataset identifier. Legacy format is ``split_subset`` such as
                ``test_database`` or ``train_queries``. Nested scenes can be referenced
                with ``scene/split_subset`` or ``scene/split`` when the directory layout is
                ``<root>/<scene>/<split>/...``.
            DATASET_DIR (str): path to the dataset directory or root folder that contains
                multiple scenes in nested sub-directories.
        """

        if "/" in name:
            scene, local_name = name.split("/", 1)
            dataset_root = os.path.join(DATASET_DIR, scene)
        else:
            dataset_root = DATASET_DIR
            local_name = name

        name_parts = local_name.split("_")
        self.split_name = name_parts[0]
        self.name = "_".join(name_parts[1:]) if len(name_parts) > 1 else None

        candidate_paths = []
        if self.name:
            candidate_paths.extend([
                os.path.join(dataset_root, self.split_name, self.name, "sequence"),
                os.path.join(dataset_root, self.split_name, self.name),
            ])
        candidate_paths.extend([
            os.path.join(dataset_root, self.split_name, "sequence"),
            os.path.join(dataset_root, self.split_name),
        ])

        for candidate in candidate_paths:
            if os.path.isdir(candidate):
                self.path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Could not resolve Robotcar path for name='{name}' under '{DATASET_DIR}'"
            )

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
