import os
from collections import OrderedDict
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Union, List

import numpy as np
import PIL.Image as Image

from .utils import CASEVPR_ROOT_DIR, AttributeDict
from .models import Flatten, L2Norm



ArrayImg = Union[np.ndarray, Image.Image]
Batchable = Union[ArrayImg, List[ArrayImg]]

def batchify(transform: transforms.Compose):
    """
    Wraps a single‐image transform so it can take:
      - single PIL or NumPy
      - list of them
      - batched NumPy array (N,H,W,C) or (N,H,W)
    Returns a tensor of shape (C,H,W) or (N,C,H,W).
    """
    def wrapper(imgs: Batchable):
        if isinstance(imgs, Image.Image):
            imgs = [imgs]

        if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            imgs_list = [Image.fromarray(np.uint8(imgs[i])) for i in range(imgs.shape[0])]

        elif isinstance(imgs, list) and isinstance(imgs[0], Image.Image):
            imgs_list = imgs
        
        elif isinstance(imgs, list) and isinstance(imgs[0], np.ndarray):
            imgs_list = [Image.fromarray(np.uint8(im)) for im in imgs]

        elif isinstance(imgs, np.ndarray) and imgs.ndim in (2, 3):
            imgs_list = [imgs]

        else:
            raise TypeError(
                f"Unsupported input type {type(imgs)} with shape "
                f"{getattr(imgs, 'shape', None)}"
            )

        # Apply the single‐image transform to each, stack if needed
        ts = [transform(im) for im in imgs_list]

        if len(ts) == 1:
            return ts[0]            # (C,H,W)
        else:
            return torch.stack(ts, 0)  # (N,C,H,W)

    return wrapper

def input_transform(img_shape=(480, 640), device=torch.device('cuda')):
    def transform(imgs, tp=None):
        """
        Args:
            imgs: Input image(s), which can be:
                - A single PIL Image
                - A single NumPy array (H, W, C) or (H, W)
                - A list of PIL Images or NumPy arrays
                - A stacked NumPy array of images (N, H, W, C) or (N, H, W)
        Returns:
            torch.Tensor: A tensor containing the transformed image(s).
        """
        # If imgs is a single image (PIL Image or NumPy array), wrap it in a list
        if isinstance(imgs, (Image.Image)):
            imgs = [imgs]
        elif isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            # Input is already a stacked NumPy array (N, H, W, C)
            pass  # We'll handle this case separately
        elif not isinstance(imgs, list):
            raise TypeError(
                "Input should be a PIL Image, NumPy array, list of images, or stacked NumPy array")

        # Check if the input is a stacked NumPy array of images
        if isinstance(imgs, np.ndarray):
            # imgs is a NumPy array with shape (N, H, W, C) or (N, H, W)
            imgs_np = imgs  # Use imgs directly
        else:
            # imgs is a list of images (PIL Images or NumPy arrays)
            if isinstance(imgs[0], np.ndarray):
                # imgs is a list of NumPy arrays
                # Shape: (N, H, W, C) or (N, H, W)
                imgs_np = np.stack(imgs, axis=0)
            elif isinstance(imgs[0], Image.Image):
                # imgs is a list of PIL Images
                # Convert each PIL Image to a NumPy array
                imgs_np = np.stack([np.array(img) for img in imgs], axis=0)
            else:
                raise TypeError(
                    "List elements must be PIL Images or NumPy arrays")

        # Convert NumPy array to torch tensor and normalize pixel values to [0, 1]
        imgs_tensor = torch.tensor(imgs_np, dtype=torch.float32, device=device)

        # Handle grayscale images
        if imgs_tensor.ndim == 4:
            # (N, H, W, C) -> (N, C, H, W)
            imgs_tensor = imgs_tensor.permute(0, 3, 1, 2)
        elif imgs_tensor.ndim == 3:
            # (N, H, W) -> (N, 1, H, W)
            imgs_tensor = imgs_tensor.unsqueeze(1)
        else:
            raise ValueError("Invalid image dimensions")

        imgs_resized = torch.nn.functional.interpolate(
            imgs_tensor, size=img_shape, mode='bilinear', antialias=True, align_corners=False,
        )

        imgs_tensor = imgs_resized / 255.0

        # Normalize images
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=device).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=device).view(1, -1, 1, 1)

        # If grayscale images, adjust mean and std
        if imgs_tensor.size(1) == 1:
            mean = mean[:, :1, :, :]
            std = std[:, :1, :, :]

        imgs_normalized = (imgs_tensor - mean) / std

        # If only one image was input, remove batch dimension
        if imgs_normalized.size(0) == 1:
            return imgs_normalized[0]
        else:
            return imgs_normalized

    return transform

def input_transform_boq(image_size=(322, 322), device=torch.device('cuda')):
    base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.to(device))
    ])
    return batchify(base)

def input_transform_vgt(image_size=(384, 384), device=torch.device('cuda')):
    base = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.to(device))
    ])
    return batchify(base)

def wrapper_print(*text, color="white"):
    print(*text)


def bytes_to_mb(x):
    return x / 1024 / 1024


class seqFrontEnd():
    def __init__(self, model_name, encoder_configs, device="cuda", tp=None, print_func=wrapper_print):
        img_encoders, seq_encoders, hvpr_encoders = encoder_configs
        self.model_name = model_name
        self.device = torch.device(device)
        self.seqlen = 1  # fake seqlen
        model_names = model_name.split("+")
        self.tp = tp
        self.print = print_func
        torch.cuda.reset_peak_memory_stats(device)
        self.init_mem = bytes_to_mb(torch.cuda.memory_allocated(device))
        if len(model_names) == 2:
            self.model_mode = "img+seq"
            img_model_name, seq_model_name = model_names
            assert img_model_name in img_encoders.keys(
            ), f"Invalid image model name {img_model_name}"
            assert seq_model_name in seq_encoders.keys(
            ), f"Invalid sequence model name {seq_model_name}"

            self.img_model_args = AttributeDict(img_encoders[img_model_name])
            self.seq_model_args = AttributeDict(seq_encoders[seq_model_name])
            self.img_model_name = img_model_name
            self.seq_model_name = seq_model_name

            self.model = self.__init_img_model(
                self.img_model_name, self.img_model_args)
            self.seq_model = self.__init_seq_model(
                self.seq_model_name, self.seq_model_args)
            self.seqlen = self.seq_model.seq_length

        elif len(model_names) == 1:
            if "hvpr" in model_name.lower():
                self.model_mode = "hvpr"
                assert model_name in hvpr_encoders.keys(
                ), f"Invalid model name {model_name}"
                self.hvpr_model_args = AttributeDict(hvpr_encoders[model_name])
                self.hvpr_model_name = model_name
                self.hvpr_model = self.__init_hvpr_model(
                    self.hvpr_model_name, self.hvpr_model_args)
                self.seqlen = self.hvpr_model.seq_length

            else:
                self.model_mode = "img"
                assert model_name in img_encoders.keys(
                ), f"Invalid model name {model_name}"
                self.img_model_args = AttributeDict(img_encoders[model_name])
                self.img_model_name = model_name
                self.model = self.__init_img_model(
                    self.img_model_name, self.img_model_args)
        else:
            raise Exception(
                f"Too many models, only support 1 or 2 models, but got {len(model_names)}")

    def __init_seq_model(self, seq_model_name, seq_model_args):
        print('===> Building model', seq_model_name)
        # Load default args if it exists
        if "class_default_args" in seq_model_args:
            class_args = seq_model_args.class_default_args
            class_args.update(seq_model_args.class_args)
            seq_model_args.class_args = class_args
        if isinstance(seq_model_args.class_args, dict):
            seq_model_args.class_args = AttributeDict(
                seq_model_args.class_args)

        seq_model = seq_model_args["class"](
            seq_model_args.class_args)
        if os.path.exists(seq_model_args.ckpt_path):
            state_dict = torch.load(
                os.path.join(CASEVPR_ROOT_DIR, seq_model_args.ckpt_path) if not seq_model_args.ckpt_path.startswith(
                    "/") else seq_model_args.ckpt_path,
                map_location=lambda storage,
                loc: storage, encoding='latin1')["model_state_dict"]
            state_dict = OrderedDict(
                {k.replace('module.', ''): v for (k, v) in state_dict.items()})
            seq_model.load_state_dict(state_dict)
        self.seq_input_transform = seq_model_args.input_transform(
            seq_model_args.img_shape, self.device)

        self.seq_pca = None
        self.pcalayer = None

        if "seq_pca_path" in seq_model_args:
            pca_path = os.path.join(CASEVPR_ROOT_DIR, seq_model_args.seq_pca_path) if not seq_model_args.seq_pca_path.startswith(
                "/") else seq_model_args.seq_pca_path
            if os.path.exists(pca_path):
                with open(pca_path, 'rb') as f:
                    self.seq_pca = pickle.load(f)
        elif "pcalayer_path" in seq_model_args:
            pca_path = os.path.join(CASEVPR_ROOT_DIR, seq_model_args.pcalayer_path) if not seq_model_args.pcalayer_path.startswith(
                "/") else seq_model_args.pcalayer_path
            pca_layer_dict = torch.load(pca_path)
            num_pcs, pool_dim, _, _ = pca_layer_dict['weight'].shape

            pca_conv = nn.Conv2d(pool_dim, num_pcs,
                                 kernel_size=(1, 1), stride=1, padding=0)
            self.pcalayer = nn.Sequential(
                *[pca_conv, Flatten(), L2Norm(dim=-1)])
            state_dict = {
                '0.weight': pca_layer_dict['weight'], '0.bias': pca_layer_dict['bias']}
            self.pcalayer.load_state_dict(state_dict)

        seq_model = seq_model.to(self.device)
        seq_model.eval()
        return seq_model

    def __init_img_model(self, img_model_name, img_model_args):
        print('===> Building model', img_model_name)
        if "class" in img_model_args:
            model = img_model_args["class"](**img_model_args.class_args)
            self.img_model_type = "class"
        elif "hub" in img_model_args:
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            model = torch.hub.load(
                img_model_args.hub["repo"], img_model_args.hub["model"], **img_model_args.hub["kwargs"])
            self.img_model_type = "hub"
        else:
            raise Exception(
                "Invalid model type, only support 'class' or 'hub'")

        if img_model_args.ckpt_path:
            ckpt_path = os.path.join(CASEVPR_ROOT_DIR, img_model_args.ckpt_path) if not img_model_args.ckpt_path.startswith(
                "/") else img_model_args.ckpt_path
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(
                    ckpt_path, map_location=lambda storage, loc: storage, encoding='latin1')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("Loaded checkpoint from {}".format(ckpt_path))
            else:
                print("Checkpoint path does not exist, using random weights")
        elif self.img_model_type == "hub":
            print(
                "No checkpoint path provided for hub model, using default weights from torch.hub")
        else:
            print("No checkpoint path provided, using random weights")

        self.input_transform = img_model_args.input_transform(
            img_model_args.img_shape, self.device)

        model = model.to(self.device)
        model.eval()
        return model

    def __init_hvpr_model(self, hvpr_model_name, hvpr_model_args):
        print('===> Building model', hvpr_model_name)
        model = hvpr_model_args["class"](**hvpr_model_args.class_args)
        ckpt_path = os.path.join(CASEVPR_ROOT_DIR, hvpr_model_args.ckpt_path) if not hvpr_model_args.ckpt_path.startswith(
            "/") else hvpr_model_args.ckpt_path
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(
                ckpt_path, map_location=lambda storage, loc: storage, encoding='latin1')

            model.load_state_dict(
                checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['state_dict'])
            print("Loaded checkpoint from {}".format(ckpt_path))
        self.input_transform = hvpr_model_args.input_transform(
            hvpr_model_args.img_shape, self.device)
        model = model.to(self.device)
        model.eval()
        return model

    def get_vector(self, img, seq=None):
        if self.model_mode == "img":
            heatmap, img_vector = self.__extract_img_feature(img)
            self.print(f"img_vector: {img_vector.shape}")
            peak_mem = bytes_to_mb(
                torch.cuda.max_memory_allocated(self.device))
            self.print(f"Memory usage: {peak_mem - self.init_mem} MB")
            return heatmap, img_vector, None
        elif self.model_mode == "img+seq":
            heatmap, img_vector = self.__extract_img_feature(img)
            seq_feature = self.__extract_seq_feature(seq)
            if seq_feature is not None:
                self.print(
                    f"img_feature: {img_vector.shape}, seq_feature: {seq_feature.shape}")
            peak_mem = bytes_to_mb(
                torch.cuda.max_memory_allocated(self.device))
            self.print(f"Memory usage: {peak_mem - self.init_mem} MB")
            return heatmap, img_vector, seq_feature
        elif self.model_mode == "hvpr":
            img_feature, seq_feature = self.__extract_hvpr_feature(img, seq)
            if seq_feature is not None:
                self.print(
                    f"img_feature: {img_feature.shape}, seq_feature: {seq_feature.shape}")
            peak_mem = bytes_to_mb(
                torch.cuda.max_memory_allocated(self.device))
            self.print(f"Memory usage: {peak_mem - self.init_mem} MB")
            return None, img_feature, seq_feature
        else:
            raise Exception("Model mode is img, cannot process sequence")

    def __extract_seq_feature(self, seq):
        if seq is not None:
            seq_tensor = self.seq_input_transform(seq)

            self.tp and self.tp.start("gen_seq_feats")
            with torch.no_grad():
                feature = self.seq_model(seq_tensor)
                feature = feature.detach().cpu().numpy()
            if self.seq_pca:
                feature = self.seq_pca.transform(feature)
            elif self.pcalayer:
                feature = self.pcalayer(torch.from_numpy(
                    feature).unsqueeze(-1).unsqueeze(-1))
                feature = feature.detach().cpu().numpy()
            self.tp and self.tp.end("gen_seq_feats")
            return feature.astype(np.float32)
        return None

    def __extract_img_feature(self, img):
        img = Image.fromarray(np.uint8(img))
        img_tensor = self.input_transform(img).unsqueeze(0)

        self.tp and self.tp.start("gen_feats")
        with torch.no_grad():
            results = self.model(img_tensor)
            if isinstance(results, tuple):
                if "boq" in self.img_model_name:
                    output, _ = results
                    heatmap = np.random.rand(30, 40)
                    output = output.data.cpu().numpy()
                else:
                    heatmap, output = results
                    output = output.data.cpu().numpy()
            else:
                heatmap = np.random.rand(30, 40)
                output = results.data.cpu().numpy()
        self.tp and self.tp.end("gen_feats")
        return heatmap, output

    def __extract_hvpr_feature(self, img, seq):
        if seq is not None:
            # self.tp and self.tp.start("input_tf")
            seq_tensor = self.input_transform(seq)
            # self.tp and self.tp.end("input_tf")
            self.tp and self.tp.start("gen_feats")
            with torch.no_grad():
                img_features, seq_feature = self.hvpr_model(seq_tensor)
                img_feature = img_features.detach(
                ).cpu().numpy()[-1][np.newaxis, :]
                seq_feature = seq_feature.detach().cpu().numpy()
            self.tp and self.tp.end("gen_feats")
            return img_feature.astype(np.float32), seq_feature.astype(np.float32)
        else:
            img = Image.fromarray(np.uint8(img))
            # img_tensor = self.input_transform(img, self.tp).unsqueeze(0).to(self.device)

            with torch.no_grad():
                img_tensor = self.input_transform(img).unsqueeze(0)
                img_feature, _ = self.hvpr_model(img_tensor, single_img=True)
                img_feature = img_feature.detach().cpu().numpy()
            return img_feature.astype(np.float32), None
