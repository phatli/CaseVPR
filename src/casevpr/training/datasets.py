"""Datasets and sampling utilities for CaseVPR sequence training."""
from __future__ import annotations

import logging
import os
import random
from glob import glob
from itertools import product
from multiprocessing import Pool, cpu_count
from os.path import join
from typing import Iterable, List, Sequence

import faiss
import numpy as np
import torch
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms

from .utils import RAMEfficient2DMatrix, extract_sequence_descriptor
from ..utils.redis_utils import Mat_Redis_Utils


__all__ = [
    "BaseDataset",
    "TrainDataset",
    "PCADataset",
    "collate_fn",
    "build_sequences",
]


def collate_fn(batch: Iterable[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine a list of samples into a batched tensor.

    Each sample is ``(images, triplets_local_indexes, triplets_global_indexes)``.
    Images already stack query, positive and negatives; here we concatenate
    across the batch and adjust local indices accordingly.
    """
    images = torch.cat([sample[0] for sample in batch])
    triplets_local_indexes = torch.cat([sample[1][None] for sample in batch])
    triplets_global_indexes = torch.cat([sample[2][None] for sample in batch])

    for i, local_indexes in enumerate(triplets_local_indexes):
        local_indexes += len(triplets_global_indexes[i]) * i

    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


def augment_sequence(sequence: List[torch.Tensor], max_step: int = 2, aug_prob: float = 0.3) -> List[torch.Tensor]:
    """Randomly duplicate frames within a sequence."""
    length = len(sequence)
    seq_to_modify = sequence[1:-1]
    available_frames = list(range(0, length))

    if random.random() < aug_prob:
        replacement = 0
        for i in range(1, length - 1):
            possible_replacements = [frame for frame in available_frames if abs(frame - i) <= max_step and frame >= i - 1 and frame >= replacement]
            if possible_replacements:
                replacement = random.choice(possible_replacements)
                seq_to_modify[i - 1] = sequence[replacement]

    return [sequence[0]] + seq_to_modify + [sequence[-1]]


def filter_nonpos_query(save_dict):
    filtered_save_dict = save_dict.copy()
    valid_qIdx = np.where(np.array([len(p) for p in save_dict['pIdx']]) > 0)[0]

    filtered_save_dict['qIdx'] = save_dict['qIdx'][valid_qIdx]
    filtered_save_dict['pIdx'] = save_dict['pIdx'][valid_qIdx]
    if len(save_dict['nonNegIdx']) > 0:
        filtered_save_dict['nonNegIdx'] = [save_dict['nonNegIdx'][i] for i in valid_qIdx]
    filtered_save_dict['q_without_pos'] = len(save_dict['q_paths']) - len(filtered_save_dict['qIdx'])

    return filtered_save_dict


def process_q(args):
    (
        q,
        q_idx_frame_to_seq,
        q_unique_idxs,
        hard_positives_per_query,
        soft_positives_per_query,
        db_idx_frame_to_seq,
        db_unique_idxs,
        split,
    ) = args

    q_without_pos = 0
    q_frame_idxs = q_idx_frame_to_seq[q]
    unique_q_frame_idxs = np.where(np.in1d(q_unique_idxs, q_frame_idxs))

    p_uniq_frame_idxs = np.unique([p for pos in hard_positives_per_query[unique_q_frame_idxs] for p in pos])

    if len(p_uniq_frame_idxs) > 0:
        lax_pIdx_seq = np.unique(
            np.where(
                np.in1d(db_idx_frame_to_seq, db_unique_idxs[p_uniq_frame_idxs]).reshape(db_idx_frame_to_seq.shape)
            )[0]
        ).astype(np.int64)
        p_uniq_frame_set = set(db_unique_idxs[p_uniq_frame_idxs])
        strict_pIdx_seq = np.array(
            [seq_idx for seq_idx, sequence in enumerate(db_idx_frame_to_seq) if set(sequence).issubset(p_uniq_frame_set)],
            dtype=np.int64,
        )
        lastframe_pIdx_seq = np.array(
            [
                seq_idx
                for seq_idx, sequence in enumerate(db_idx_frame_to_seq)
                if sequence[-1] in hard_positives_per_query[unique_q_frame_idxs[0][-1]]
            ],
            dtype=np.int64,
        )

        if split == 'train':
            nonNeg_uniq_frame_idxs = np.unique(
                [p for pos in soft_positives_per_query[unique_q_frame_idxs] for p in pos]
            )
            lax_nonNegIdx = np.unique(
                np.where(
                    np.in1d(db_idx_frame_to_seq, db_unique_idxs[nonNeg_uniq_frame_idxs]).reshape(db_idx_frame_to_seq.shape)
                )[0]
            ).astype(np.int64)
            nonNeg_uniq_frame_set = set(db_unique_idxs[nonNeg_uniq_frame_idxs])
            strict_nonNegIdx = np.array(
                [seq_idx for seq_idx, sequence in enumerate(db_idx_frame_to_seq) if set(sequence).issubset(nonNeg_uniq_frame_set)],
                dtype=np.int64,
            )
        else:
            lax_nonNegIdx = None
            strict_nonNegIdx = None

    else:
        q = None
        lax_pIdx_seq = None
        strict_pIdx_seq = None
        lastframe_pIdx_seq = None
        lax_nonNegIdx = None
        strict_nonNegIdx = None
        q_without_pos = 1

    return (
        q,
        lax_pIdx_seq,
        strict_pIdx_seq,
        lastframe_pIdx_seq,
        lax_nonNegIdx,
        strict_nonNegIdx,
        q_without_pos,
    )


class BaseDataset(data.Dataset):
    def __init__(
        self,
        cities: Sequence[str] | str = "",
        dataset_folder: str = "datasets",
        split: str = "train",
        base_transform: transforms.Compose | None = None,
        seq_len: int = 3,
        pos_thresh: int = 25,
        neg_thresh: int = 25,
        cut_last_frame: bool = False,
        reverse_frames: bool = False,
        seq_gt_strategy: str = "lax",
        neg_seq_gt_strategy: str = "lax",
    ) -> None:
        super().__init__()
        self.dataset_folder = join(dataset_folder, split)
        self.seq_len = seq_len + (1 if cut_last_frame else 0)
        self.base_transform = base_transform

        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        self.seq_gt_strategy = seq_gt_strategy
        self.neg_seq_gt_strategy = neg_seq_gt_strategy
        self.init_data(cities, split, pos_thresh, neg_thresh)

        if reverse_frames:
            self.db_paths = [",".join(path.split(',')[::-1]) for path in self.db_paths]

        self.images_paths = self.db_paths + self.q_paths
        self.database_num = len(self.db_paths)
        self.queries_num = len(self.qIdx)
        self.redis_handle = Mat_Redis_Utils()

        if cut_last_frame:
            self._cut_last_frame()

    def init_data(self, cities, split, pos_thresh, neg_thresh):
        if cities != '' and not isinstance(cities, list):
            cities = [cities]

        if 'msls' in self.dataset_folder and split == 'train':
            cache_file = f'cache/msls_seq{self.seq_len}_{cities}_pos{pos_thresh}_neg{neg_thresh}.torch'
        else:
            cache_file = f'cache/{self.dataset_folder.split("/")[-2]}_seq{self.seq_len}_{split}_pos{pos_thresh}_neg{neg_thresh}.torch'

        if os.path.isfile(cache_file):
            try:
                logging.info("Loading cached data from %s", cache_file)
                cache_dict = torch.load(cache_file)
                self._load_dict(cache_dict, split)
                return
            except Exception as exc:
                logging.error("Failed to load cache %s: %s", cache_file, exc)
        else:
            os.makedirs('cache', exist_ok=True)
            logging.info('Data structures not cached, building them now...')

        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")

        self.db_paths, all_db_paths, db_idx_frame_to_seq = build_sequences(
            database_folder,
            seq_len=self.seq_len,
            cities=cities,
            desc='loading database...'
        )
        self.q_paths, all_q_paths, q_idx_frame_to_seq = build_sequences(
            queries_folder,
            seq_len=self.seq_len,
            cities=cities,
            desc='loading queries...'
        )

        q_unique_idxs = np.unique([idx for seq_frames_idx in q_idx_frame_to_seq for idx in seq_frames_idx])
        db_unique_idxs = np.unique([idx for seq_frames_idx in db_idx_frame_to_seq for idx in seq_frames_idx])

        database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in all_db_paths[db_unique_idxs]]
        ).astype(np.float64)
        queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in all_q_paths[q_unique_idxs]]
        ).astype(np.float64)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(database_utms)
        hard_positives_per_query = knn.radius_neighbors(queries_utms, radius=pos_thresh, return_distance=False)

        if split == 'train':
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(database_utms)
            soft_positives_per_query = knn.radius_neighbors(queries_utms, radius=neg_thresh, return_distance=False)
        else:
            soft_positives_per_query = None

        qIdx = []
        lax_pIdx = []
        strict_pIdx = []
        lastframe_pIdx = []
        lax_nonNegIdx = []
        strict_nonNegIdx = []
        q_without_pos_list = []
        total_tasks = len(q_idx_frame_to_seq)

        with Pool() as pool:
            for result in pool.imap(
                process_q,
                [
                    (
                        q,
                        q_idx_frame_to_seq,
                        q_unique_idxs,
                        hard_positives_per_query,
                        soft_positives_per_query,
                        db_idx_frame_to_seq,
                        db_unique_idxs,
                        split,
                    )
                    for q in range(total_tasks)
                ],
                chunksize=cpu_count(),
            ):
                q_ind, l_pIdx, s_pIdx, lf_pIdx, l_nNegIdx, s_nNegIdx, q_w_o_pos = result
                if q_ind is not None:
                    qIdx.append(q_ind)
                if l_pIdx is not None:
                    lax_pIdx.append(l_pIdx)
                if s_pIdx is not None:
                    strict_pIdx.append(s_pIdx)
                if lf_pIdx is not None:
                    lastframe_pIdx.append(lf_pIdx)
                if l_nNegIdx is not None:
                    lax_nonNegIdx.append(l_nNegIdx)
                if s_nNegIdx is not None:
                    strict_nonNegIdx.append(s_nNegIdx)
                if q_w_o_pos is not None:
                    q_without_pos_list.append(q_w_o_pos)

        self.qIdx = np.array(qIdx)
        self.lax_pIdx = np.array(lax_pIdx, dtype=object)
        self.strict_pIdx = np.array(strict_pIdx, dtype=object)
        self.lastframe_pIdx = np.array(lastframe_pIdx, dtype=object)
        self.lax_nonNegIdx = lax_nonNegIdx
        self.strict_nonNegIdx = strict_nonNegIdx
        self.q_without_pos = sum(q_without_pos_list)

        save_dict = {
            'db_paths': self.db_paths,
            'q_paths': self.q_paths,
            'qIdx': self.qIdx,
            'lax_nonNegIdx': self.lax_nonNegIdx,
            'strict_nonNegIdx': self.strict_nonNegIdx,
            'q_without_pos': self.q_without_pos,
            'lax_pIdx': self.lax_pIdx,
            'strict_pIdx': self.strict_pIdx,
            'lastframe_pIdx': self.lastframe_pIdx,
        }
        torch.save(save_dict, cache_file)
        self._load_dict(save_dict, split)

    def _load_dict(self, load_dict, split):
        if self.seq_gt_strategy == "lax":
            load_dict["pIdx"] = load_dict["lax_pIdx"]
        elif self.seq_gt_strategy == "strict":
            load_dict["pIdx"] = load_dict["strict_pIdx"]
        elif self.seq_gt_strategy == "lastframe":
            load_dict["pIdx"] = load_dict["lastframe_pIdx"]
        else:
            raise ValueError(f"Invalid sequence ground truth strategy: {self.seq_gt_strategy}")

        if split == 'train':
            if self.neg_seq_gt_strategy == "lax":
                load_dict["nonNegIdx"] = load_dict["lax_nonNegIdx"]
            else:
                load_dict["nonNegIdx"] = load_dict["strict_nonNegIdx"]
        else:
            load_dict["nonNegIdx"] = []

        load_dict = filter_nonpos_query(load_dict)
        self.__dict__.update(load_dict)

    def _cut_last_frame(self):
        for attr in ("images_paths", "db_paths", "q_paths"):
            paths = getattr(self, attr)
            for i, seq in enumerate(paths):
                paths[i] = ','.join(seq.split(',')[:-1])

    def __getitem__(self, index):
        old_index = index
        if index >= self.database_num:
            q_index = index - self.database_num
            index = self.qIdx[q_index] + self.database_num

        tensor_seq = []
        for im in self.images_paths[index].split(','):
            tensor_seq.append(self.base_transform(self.redis_handle.load_PIL(join(self.dataset_folder, im))))
        img = torch.stack(tensor_seq)

        return img, index, old_index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self) -> str:
        return f"< {self.__class__.__name__}, ' #database: {self.database_num}; #queries: {self.queries_num} >"

    def get_positives(self):
        return self.pIdx


def filter_by_cities(path: str, cities: Sequence[str]) -> bool:
    return any(path.find(city) > 0 for city in cities)


def build_sequences(folder: str, seq_len: int = 3, cities: Sequence[str] | str = "", desc: str = 'loading'):
    if cities != '' and not isinstance(cities, list):
        cities = [cities]
    base_path = os.path.dirname(folder)
    paths: List[str] = []
    all_paths: List[str] = []
    idx_frame_to_seq = []
    seqs_folders = sorted(glob(join(folder, '*'), recursive=True))

    for seq in seqs_folders:
        start_index = len(all_paths)
        frame_nums = np.array(list(map(lambda x: int(x.split('@')[4]), sorted(glob(join(seq, '*'))))))
        full_seq_paths = sorted(glob(join(seq, '*')))
        seq_paths = np.array([s_p.replace(f'{base_path}/', '') for s_p in full_seq_paths])

        if cities:
            sample_path = seq_paths[0]
            if not filter_by_cities(sample_path, cities):
                continue

        sorted_idx_frames = np.argsort(frame_nums)
        all_paths += list(seq_paths[sorted_idx_frames])
        half = seq_len // 2
        for idx, frame_num in enumerate(frame_nums):
            if idx < half or idx >= (len(frame_nums) - half):
                continue

            if seq_len % 2 == 0:
                offsets = np.arange(-half, half) + idx
            else:
                offsets = np.arange(-half, half + 1) + idx

            if offsets[0] < 0 or offsets[-1] >= len(frame_nums):
                continue

            seq_idx = offsets
            consecutive_frames = frame_nums[sorted_idx_frames][seq_idx]
            if len(consecutive_frames) == seq_len and (np.diff(consecutive_frames) == 1).all():
                paths.append(",".join(seq_paths[sorted_idx_frames][seq_idx]))
                idx_frame_to_seq.append(seq_idx + start_index)

    return paths, np.array(all_paths), np.array(idx_frame_to_seq)


class TrainDataset(BaseDataset):
    def __init__(
        self,
        cities: Sequence[str] | str = "",
        dataset_folder: str = "datasets",
        split: str = "train",
        base_transform: transforms.Compose | None = None,
        seq_len: int = 3,
        pos_thresh: int = 25,
        neg_thresh: int = 25,
        infer_batch_size: int = 8,
        n_gpus: int = 1,
        features_dim: int = 256,
        img_shape: Sequence[int] = (480, 640),
        cut_last_frame: bool = False,
        cached_negatives: int = 1000,
        cached_queries: int = 1000,
        nNeg: int = 10,
        seq_gt_strategy: str = "lax",
        neg_seq_gt_strategy: str = "lax",
        augseq: bool = False,
        augseq_max_step: int = 2,
        augseq_prob: float = 0.3,
        seq_neg_mining: bool = False,
        seq_neg_mining_prob: float = 0.3,
    ) -> None:
        super().__init__(
            dataset_folder=dataset_folder,
            split=split,
            cities=cities,
            base_transform=base_transform,
            seq_len=seq_len,
            pos_thresh=pos_thresh,
            neg_thresh=neg_thresh,
            cut_last_frame=cut_last_frame,
            seq_gt_strategy=seq_gt_strategy,
            neg_seq_gt_strategy=neg_seq_gt_strategy,
        )
        self.cached_negatives = cached_negatives
        self.cached_queries = cached_queries
        self.n_gpus = n_gpus
        self.num_workers = 2 * max(int(n_gpus), 1)
        self.device = torch.device('cuda' if n_gpus > 0 else 'cpu')
        self.features_dim = features_dim
        self.bs = infer_batch_size
        self.img_shape = img_shape
        self.nNeg = nNeg
        self.is_inference = False
        self.query_transform = self.base_transform
        self.augseq = augseq
        self.augseq_max_step = augseq_max_step
        self.augseq_prob = augseq_prob
        self.seq_neg_mining = seq_neg_mining
        self.seq_neg_mining_prob = seq_neg_mining_prob
        self.redis_handle = Mat_Redis_Utils()

    def __getitem__(self, index):
        if self.is_inference:
            return super().__getitem__(index)
        query_index, best_positive_index, neg_indexes = torch.split(
            self.triplets_global_indexes[index], (1, 1, self.nNeg)
        )

        def load_sequence(seq_paths: str) -> List[torch.Tensor]:
            return [
                self.base_transform(self.redis_handle.load_PIL(join(self.dataset_folder, im)))
                for im in seq_paths.split(',')
            ]

        if self.augseq:
            query = torch.stack(
                augment_sequence(
                    load_sequence(self.q_paths[query_index]),
                    max_step=self.augseq_max_step,
                    aug_prob=self.augseq_prob,
                )
            )
        else:
            query = torch.stack(load_sequence(self.q_paths[query_index]))

        positive = torch.stack(load_sequence(self.db_paths[best_positive_index]))

        negatives = [torch.stack(load_sequence(self.db_paths[idx])) for idx in neg_indexes]
        if self.seq_neg_mining and random.random() < self.seq_neg_mining_prob:
            mined_sequence = self._mine_seq_hard_neg(best_positive_index)
            if mined_sequence is not False and len(mined_sequence) > 0:
                replace_idx = random.randint(0, self.nNeg - 1)
                negatives[replace_idx] = torch.stack(
                    [
                        self.base_transform(self.redis_handle.load_PIL(join(self.dataset_folder, im)))
                        for im in mined_sequence
                    ]
                )

        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat(
                (triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3))
            )
        return images, triplets_local_indexes, self.triplets_global_indexes[index]

    def __len__(self):
        if self.is_inference:
            return super().__len__()
        return len(self.triplets_global_indexes)

    def compute_triplets(self, model: torch.nn.Module) -> None:
        self.is_inference = True
        self.compute_triplets_partial(model)
        self.is_inference = False

    def compute_cache(self, model, subset_ds, cache_shape):
        subset_dl = DataLoader(
            dataset=subset_ds,
            num_workers=self.num_workers,
            batch_size=self.bs,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
        )
        model = model.eval()
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        with torch.no_grad():
            for images, indexes, _ in subset_dl:
                target_indexes = indexes.cpu().numpy()
                images = images.view(-1, 3, self.img_shape[0], self.img_shape[1])
                if images.shape[0] % self.seq_len != 0:
                    raise ValueError(
                        f"Sequence batch shape mismatch: total_frames={images.shape[0]}, "
                        f"expected multiples of seq_len={self.seq_len}"
                    )
                if (images.shape[0] % (self.seq_len * self.n_gpus) != 0) and self.n_gpus > 1:
                    model.module = model.module.to('cuda:1')
                    for sequence in range(images.shape[0] // self.seq_len):
                        n_seq = sequence * self.seq_len
                        seq_images = images[n_seq: n_seq + self.seq_len].to('cuda:1')
                        outputs = model.module(seq_images)
                        seq_features = extract_sequence_descriptor(outputs)
                        cache[target_indexes[sequence], :] = seq_features.cpu().numpy()
                    model = model.cuda()
                else:
                    outputs = model(images.to(self.device))
                    features = extract_sequence_descriptor(outputs)
                    cache[target_indexes] = features.cpu().numpy()
        return cache

    def get_best_positive_index(self, qidx, cache, query_features):
        if query_features.ndim != 1 or query_features.shape[0] != self.features_dim:
            raise ValueError(
                f"Query features shape mismatch: expected ({self.features_dim},) but got {query_features.shape}"
            )
        positives_features = cache[self.pIdx[qidx]]
        try:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.GpuIndexFlatL2(res, self.features_dim)
            faiss_index.add(positives_features)
        except RuntimeError as err:
            if "cudaMalloc error out of memory" in str(err):
                faiss_index = faiss.IndexFlatL2(self.features_dim)
                faiss_index.add(positives_features)
            else:
                raise
        _, neighbors = faiss_index.search(query_features.reshape(1, -1), 1)
        return self.pIdx[qidx][neighbors[0][0]]

    def get_hardest_negatives_indexes(self, cache, query_features, neg_indexes):
        negatives_features = cache[neg_indexes]
        try:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.GpuIndexFlatL2(res, self.features_dim)
            faiss_index.add(negatives_features)
        except RuntimeError as err:
            if "cudaMalloc error out of memory" in str(err):
                faiss_index = faiss.IndexFlatL2(self.features_dim)
                faiss_index.add(negatives_features)
            else:
                raise
        _, negatives = faiss_index.search(query_features.reshape(1, -1), self.nNeg)
        return neg_indexes[negatives[0]]

    def compute_triplets_partial(self, model):
        import psutil

        logging.debug(
            "Cache usage: current RAM %s%%, dataset cache size %d",
            psutil.virtual_memory().percent,
            self.cached_queries,
        )

        sampled_queries_indexes = np.random.choice(self.queries_num, size=self.cached_queries, replace=False)
        sampled_database_indexes = np.random.choice(self.database_num, size=self.cached_negatives, replace=False)

        positives_indexes = np.unique(
            [idx for db_idx in self.pIdx[sampled_queries_indexes] for idx in db_idx]
        )
        subset_indices = list(sampled_database_indexes) + list(positives_indexes) + list(
            sampled_queries_indexes + self.database_num
        )
        subset_ds = Subset(self, subset_indices)

        cache = self.compute_cache(
            model, subset_ds, cache_shape=(len(self), self.features_dim)
        )

        self.triplets_global_indexes = []
        for q in sampled_queries_indexes:
            qidx = self.qIdx[q] + self.database_num
            query_features = cache[qidx]
            best_positive_index = self.get_best_positive_index(q, cache, query_features)
            if isinstance(best_positive_index, np.ndarray):
                best_positive_index = best_positive_index[0]

            soft_positives = self.nonNegIdx[q]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            neg_indexes = self.get_hardest_negatives_indexes(cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((self.qIdx[q], best_positive_index, *neg_indexes))

        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)

    def _mine_seq_hard_neg(self, best_positive_index):
        best_positive_sequence = self.db_paths[best_positive_index].split(',')
        gps_coords = [self._extract_gps_coords(im) for im in best_positive_sequence]
        length = len(best_positive_sequence)

        for i in range(length - 1):
            if self._are_gps_coords_close(gps_coords[i], gps_coords[i + 1]):
                return False

        original_indices = list(range(length))
        shuffled_indices = original_indices.copy()

        def is_order_changed(original, shuffled):
            return original != shuffled

        while not is_order_changed(original_indices, shuffled_indices):
            random.shuffle(shuffled_indices)

        return [best_positive_sequence[idx] for idx in shuffled_indices]

    def _extract_gps_coords(self, image_path):
        parts = image_path.split("@")
        if len(parts) < 3:
            raise ValueError(f"Invalid image path format for GPS extraction: {image_path}")
        latitude, longitude = map(float, parts[1:3])
        return latitude, longitude

    def _are_gps_coords_close(self, coord1, coord2, threshold=0.5):
        coord1 = np.array(coord1)
        coord2 = np.array(coord2)
        distance = np.linalg.norm(coord1 - coord2)
        return distance < threshold


class PCADataset(data.Dataset):
    def __init__(
        self,
        cities: Sequence[str] | str = "",
        dataset_folder: str = "dataset",
        split: str = "val",
        base_transform: transforms.Compose | None = None,
        seq_len: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Folder {dataset_folder} does not exist.")
        self.base_transform = base_transform
        self.redis_handle = Mat_Redis_Utils()

        if 'robotcar_ori' in dataset_folder:
            self.dataset_folder = os.path.join(dataset_folder, split)
            folders = list(product(['train', 'val'], ['queries', 'database'])) + [('test', 'database')]
            self.db_paths = []
            for folder in folders:
                split_name, subset = folder
                load_folder = join(dataset_folder, split_name, subset)
                paths, _, _ = build_sequences(
                    load_folder,
                    seq_len=self.seq_len,
                    cities=cities,
                    desc="Loading database to compute PCA...",
                )
                self.db_paths += paths
        else:
            self.dataset_folder = join(dataset_folder, split)
            database_folder = join(self.dataset_folder, "queries")
            self.db_paths, _, _ = build_sequences(
                database_folder,
                seq_len=self.seq_len,
                cities=cities,
                desc="Loading database to compute PCA...",
            )

        self.db_num = len(self.db_paths)

    def __getitem__(self, index):
        tensors = [
            self.base_transform(self.redis_handle.load_PIL(os.path.join(self.dataset_folder, path)))
            for path in self.db_paths[index].split(',')
        ]
        return torch.stack(tensors)

    def __len__(self):
        return self.db_num

    def __repr__(self) -> str:
        return f"< {self.__class__.__name__}, ' #database: {self.db_num} >"
