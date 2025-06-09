from os import makedirs
from os.path import exists, join
from scipy import spatial
import numpy as np
import cv2
import torch
import pickle
from collections import deque
import faiss
from .utils import CASEVPR_ROOT_DIR


seqbackend_params = {
    'backend_name': "seqslam",
    'ds': "5",
    'thresh_out': "100",
    'stp_gap': "14",
    'matching_K': "5",
    'matching_KK': "4"
}


def getRigidT(P, Q, use_scale_factor=False):
    """
        This function is uesd to get Rigid Transform matrix of two point cloud P and Q
        Q = P*cR + t
        return c,R,t
    """
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)
    if use_scale_factor:
        varP = np.var(P, axis=0).sum()
        c = 1/varP * np.sum(S)  # scale factor
        t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)
    else:
        t = Q.mean(axis=0) - P.mean(axis=0).dot(R)
        c = 1
    return c, R, t


class seqBackEnd():
    """
    Class representing the backend for sequence-based VPR.

    Args:
        imgDB_shape (tuple, optional): Shape of the image database. Defaults to (224*720//1280, 224, 3).
        frontend_name (str, optional): Name of the frontend. Defaults to "applar".
        frontend_seqlen (int, optional): Frontend model's sequence length. Defaults to 0.
        dataset_name (str, optional): Name of the reference dataset. Defaults to 'carparkA'.
        dataset2_name (str, optional): Name of the query dataset. Defaults to 'carparkA'.
        ds (int, optional): Distance of sequence history. Defaults to 14.
        thresh_out (float, optional): Output threshold for first method. Defaults to None.
        thresh_out2 (float, optional): Output threshold for second method. Defaults to None.
        matching_K (int, optional): Number of templates being searched. Defaults to 5.
        matching_KK (int, optional): Number of neighbor being searched. Defaults to 4.
        stp_gap (int, optional): Force starting point gap in column-wise. Defaults to 14.
        backend_name (str, optional): Name of the backend. Defaults to "adapt".
        positive_dist (int, optional): Positive distance to calculate groundtruth. Defaults to 11.
        print_func (function, optional): Print function. Defaults to print.
        save_path (str, optional): Path to save the output. Defaults to join(CASEVPR_ROOT_DIR, "output", "ckpts").
        image_dir (str, optional): Directory to store temporary images. Defaults to join(CASEVPR_ROOT_DIR, "output", "IMG_TEMP").
        dist_cos_thres (float, optional): Cosine distance threshold to keep the frame as keyframe. Defaults to 0.7.
        seq_gt (bool, optional): Whether to calculate sequence groundtruth. Defaults to False.
        seq_gt_vgt (bool, optional): Whether to calculate sequence groundtruth as VGT. Defaults to False.
        save_seq_mode (str, optional): Mode for saving sequence images. Defaults to "selected".
        from_scratch (bool, optional): Flag indicating whether current run is started from scratch. Defaults to True.
        sim (str, optional): Similarity metric. Defaults to "cos".
        tp (None, optional): TimeProbe instance for time profiling. Defaults to None.
        search_nonkeyframe (bool, optional): Whether to search non-keyframe. Defaults to False.
        nosearch_range (int, optional): Range that won't be searched in the last of the vectors_database. Defaults to None.
        only_log_vectors (bool, optional): Only log vectors of new frame, do not update difference matrix and search. Defaults to False.
    """

    def __init__(self, imgDB_shape=(224*720//1280, 224, 3), frontend_name="netvlad", frontend_seqlen=0, dataset_name='carparkA', dataset2_name='carparkA', ds=14, thresh_out=100, thresh_out2=None,  matching_K=5, matching_KK=4, stp_gap=14, backend_name="seqslam", positive_dist=11, print_func=print, save_path=join(CASEVPR_ROOT_DIR, "output", "ckpts"), image_dir=join(CASEVPR_ROOT_DIR, "output", "IMG_TEMP"), dist_cos_thres=0.7, seq_gt=False, seq_gt_vgt=False, save_seq_mode="selected", from_scratch=True, sim="cos", tp=None, search_nonkeyframe=False, nosearch_range=None, min_search_range=None, only_log_vectors=False):
        self.timestamps = []
        self.coordinates = []
        self.gt_idx = []
        self.vectors_database = None
        self.vectors_cache = None
        self.D = None
        self.DD = None
        self.DD_enhanced = None
        self.sim = sim
        self.retrieved_seq_indices = None
        self.retrieved_seq_D = None
        self.positive_dist = int(positive_dist)
        self.frontend_name = frontend_name
        self.frontend_seqlen = int(frontend_seqlen)
        self.global_min_value = 100
        self.search_range = None
        self.fix_search_range = False
        self.dbFeats_fix = None
        self.backend_name = backend_name
        self.print = print_func
        self.tp = tp
        self.from_scratch = True
        self.idx = 0
        self.local_idx = 0
        self.seq_gt = seq_gt
        self.seq_gt_vgt = seq_gt_vgt
        self.from_scratch = from_scratch
        self.replace_mode = False
        self.thresh_out = float(thresh_out)

        if "+" in backend_name:
            if not thresh_out2:
                self.thresh_out2 = self.thresh_out

        self.dist_cos_thres = dist_cos_thres
        self.imgDB_shape = imgDB_shape

        filepath1 = join(image_dir, dataset_name)
        filepath2 = join(image_dir, dataset2_name)

        self.querypath = filepath1 if self.from_scratch else filepath2
        self.dbpath = filepath1

        if not exists(filepath1):
            makedirs(filepath1)
        if not exists(filepath2):
            makedirs(filepath2)

        self.save_path = save_path
        if not exists(self.save_path):
            makedirs(self.save_path)
        self.dataset_name = dataset_name
        self.dataset2_name = dataset2_name
        self.saved_file = join(
            self.save_path, f"{self.frontend_name}-distcos{(str(self.dist_cos_thres)).replace('.','')}-{self.dataset_name}.pkl")

        # * ANCHOR parameters for Adapt-seqSLAM
        self.matching_ds = int(ds) if backend_name != "none" else 1
        self.nosearch_range = int(
            ds) if nosearch_range is None else int(nosearch_range)
        self.min_search_range = 2 * \
            int(ds) if min_search_range is None else int(min_search_range)
        # how many templates are being searched
        self.matching_K = int(matching_K) if backend_name != "none" else 20
        # how far neighbor are being searched
        self.matching_KK = int(matching_KK)
        # force starting point gap in colomn-wise
        self.stp_gap = int(stp_gap)

        # * ANCHOR parameters for SeqSLAM
        self.matching_vmin = 0.8
        self.matching_vmax = 1.2
        self.posP = None

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if backend_name == "none":
            self.is_positive_1 = []
            self.is_positive_5 = []
            self.is_positive_10 = []
            self.is_positive_20 = []

        # * ANCHOR sequence descriptors backend
        if self.__isSeqDesc():
            self.stacked_images = None  # add in addKeyFrame()
            self.seqID_to_imgID = []
            self.seq_vectors = None
            self.possible_blocks = None
            self.retrieved_img_indices_seq = None

        # * ANCHOR parameters for save seq
        self.save_seq_mode = save_seq_mode
        self.save_seq = False

        # * ANCHOR parameters for search non-keyframe
        self.search_nonkeyframe = search_nonkeyframe
        self.last_keyframe_flag = True
        self.current_keyframe_flag = True

        # * ANCHOR parameters for seq search
        self.seq_index = None

        # * ANCHOR parameters for GT faiss
        self.coords_index = None

        # * ANCHOR Only log vectors mode, no search and no update D matrix
        self.only_log_vectors = only_log_vectors

    def saveCheckpoint(self, **external_kwargs):
        ckpt = {
            "timestamps": self.timestamps,
            "coordinates": self.coordinates,
            "vectors_cache": np.array(self.vectors_cache),
            "vectors_database": np.array(self.vectors_database),
            "global_min": self.global_min_value
        }
        if self.__isSeqDesc():
            ckpt.update({
                'stacked_images': np.array(self.stacked_images),
                'seq_vectors': np.array(self.seq_vectors),
                'seqID_to_imgID': self.seqID_to_imgID
            })
        ckpt.update(external_kwargs)

        with open(self.saved_file, 'wb') as f:
            pickle.dump(ckpt, f)

    def loadCheckpoint(self, freeze_search_range=True, seperate_ds=False):
        """load checkpoint

        Args:
            freeze_search_range (bool, optional): Whether to freeze search range withtin checkpoint range. Defaults to True.
            seperate_ds (bool, optional): Whether dataset is seperated in loop1 and loop2

        """
        self.from_scratch = False
        with open(self.saved_file, 'rb') as f:
            ckpt = pickle.load(f)
        self.coordinates = ckpt['coordinates']
        self.__init_coords_index(add_coords=True)
        self.timestamps = ckpt['timestamps']
        self.vectors_cache = self.__loadSavedArrayToDeque(
            ckpt["vectors_cache"])
        self.vectors_database = self.__loadSavedArrayToList(
            ckpt["vectors_database"])
        self.idx = len(self.timestamps)
        self.__correctVectorsCache()
        self.D = self.__getDifferenceMatrix(
            self.vectors_cache, self.vectors_database)
        if freeze_search_range and not seperate_ds:
            self.search_range = len(self.vectors_database)
            self.fix_search_range = True

        if self.__isSeqDesc():
            if not seperate_ds:
                self.stacked_images = self.__loadSavedArrayToDeque(
                    ckpt['stacked_images'])
            self.seqID_to_imgID = ckpt['seqID_to_imgID']
            self.seq_vectors = self.__loadSavedArrayToList(ckpt['seq_vectors'])
            if freeze_search_range:
                self.seq_search_range = len(self.seq_vectors)

        internal_keys = ["coordinates", "timestamps", "vectors_cache", "vectors_database", "global_min", "stacked_images", "seq_vectors", "seqID_to_imgID"]
        external_ckpt = {k: v for k, v in ckpt.items() if k not in internal_keys}
        return external_ckpt


    def addKeyFrame(self, timestamp, img_feature, image, coordinate, seq_feature=None, no_search=False):
        """
        This function is used to process each frame, filtering out the non-key frame

        Args:
            timestamp (string): timestamp of the input frame
            img_feature (dnarray): feature descriptors of the input frame
            image (dnarray): input frame
            coordinate (tuple): raw coordinate of the input frame

        Returns:
            tuple: (retrieved_id, is_positive)
        """
        # self.print('current coord',coordinate)
        self.tp and self.tp.start("addKeyFrame")
        # * ANCHOR check similar with last frame or not
        self.last_keyframe_flag = self.current_keyframe_flag
        self.replace_mode = self.search_nonkeyframe and not self.last_keyframe_flag

        if type(coordinate) == np.ndarray:
            coordinate = coordinate.tolist()

        if self.vectors_cache is not None:
            # if last frame is keyframe, compare with last frame, else compare with last keyframe

            cache_vector_last = self.vectors_cache[-2 if self.replace_mode and len(
                self.vectors_cache) > 1 else -1][np.newaxis, :]
            self.current_keyframe_flag = self.__isKeyframe(
                img_feature, cache_vector_last, self.dist_cos_thres)

            if self.current_keyframe_flag:
                if self.replace_mode:
                    # last frame is not a keyframe, but current frame is a keyframe, don't need to add up idx cause it's already added
                    self.vectors_cache[-1] = img_feature.squeeze()
                else:
                    # last frame is a keyframe, and current frame is a keyframe
                    self.idx += 1
                    self.local_idx += 1
                    self.vectors_cache.append(img_feature.squeeze())
            else:
                if not self.search_nonkeyframe:
                    # not a keyframe and abandon

                    self.tp and self.tp.end("addKeyFrame")
                    return 'NOT A KEY FRAME!', False
                if self.replace_mode:
                    self.vectors_cache[-1] = img_feature.squeeze()
                else:
                    # not a keyframe, and last frame is a keyframe, perform a search
                    self.vectors_cache.append(img_feature.squeeze())
                    self.idx += 1
                    self.local_idx += 1

        else:
            self.vectors_cache = deque([img_feature.squeeze()])
            self.idx += 1
            self.local_idx += 1

        # * ANCHOR Stack keyframes in sequence descriptors
        if self.__isSeqDesc():
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.stacked_images is None:
                self.stacked_images = deque([img])
            elif len(self.stacked_images) < self.frontend_seqlen - 1:
                if self.replace_mode:
                    self.stacked_images[-1] = img
                else:
                    self.stacked_images.append(img)
            else:
                if self.replace_mode:
                    self.stacked_images[-1] = img
                else:
                    self.stacked_images.append(img)
                    self.stacked_images_pop_cache = self.stacked_images.popleft()

                if self.seq_vectors is None:
                    assert seq_feature is not None, "No sequence feature input"
                    # self.seq_vectors = seq_feature
                    self.seq_vectors = [seq_feature.squeeze()]
                    # self.seqID_to_imgID.append(len(self.timestamps) - 1)
                    self.__appendOrReplace(self.seqID_to_imgID, len(
                        self.seq_vectors) - 1, self.idx - 1)
                else:
                    self.seq_vectors.append(seq_feature.squeeze())
                    # self.seqID_to_imgID.append(len(self.timestamps) - 1)
                    self.__appendOrReplace(self.seqID_to_imgID, len(
                        self.seq_vectors) - 1, self.idx - 1)
                    # here should not -1 because current timestamp haven't append yet therefore should be len(self.timestamps) - 1 + 1

        # * ANCHOR Save the key frame as a png file
        filename = join(self.querypath, f"{timestamp}.png")
        image = cv2.resize(image, dsize=(
            self.imgDB_shape[1], self.imgDB_shape[0]))
        cv2.imwrite(filename, image)

        if self.replace_mode:
            is_vectors_cache_full = len(self.vectors_cache) >= self.matching_ds
        else:
            is_vectors_cache_full = len(self.vectors_cache) > self.matching_ds
        # self.print('len(vectors-cache', len(self.vectors_cache))
        if not is_vectors_cache_full:
            self.__appendOrReplace(self.timestamps, self.idx - 1, timestamp)
            self.__appendOrReplace(self.coordinates, self.idx - 1, coordinate)
            self.__addOrReplaceCoordIndex(self.idx - 1, coordinate)

            self.tp and self.tp.end("addKeyFrame")
            return 'NO LOOP DETECTED!', False
        if self.vectors_database is None:
            if len(self.vectors_cache) > self.matching_ds:
                self.vectors_database = [self.vectors_cache.popleft()]
            self.__appendOrReplace(self.timestamps, self.idx - 1, timestamp)
            self.__appendOrReplace(self.coordinates, self.idx - 1, coordinate)
            self.__addOrReplaceCoordIndex(self.idx - 1, coordinate)

            self.tp and self.tp.end("addKeyFrame")
            return 'NO LOOP DETECTED!', False
        if len(self.vectors_cache) > self.matching_ds:
            self.vectors_database.append(self.vectors_cache.popleft())
        self.__appendOrReplace(self.timestamps, self.idx - 1, timestamp)
        self.__appendOrReplace(self.coordinates, self.idx - 1, coordinate)
        self.__addOrReplaceCoordIndex(self.idx - 1, coordinate)

        self.print(
            f'On {self.dataset_name} vs {self.dataset2_name} using {self.frontend_name}_{self.backend_name}' if not self.from_scratch else f'First loop on {self.dataset_name} using {self.frontend_name}', color="blue")
        self.print(
            f"Current idx:{self.idx - 1}, local_idx:{self.local_idx}, it is a {'keyframe' if self.current_keyframe_flag else 'non-keyframe'}.")

        if len(self.vectors_database) < self.min_search_range + self.nosearch_range:
            # ensure meet the minimal requirements of sequence retrieval

            self.tp and self.tp.end("addKeyFrame")
            return 'NO LOOP DETECTED!', False

        if self.only_log_vectors:
            self.tp and self.tp.end("addKeyFrame")
            return 'NO LOOP DETECTED!', False
        self.__updateGTIdx()
        if no_search:
            self.D = self.__updateDifferenceMatrix(
                self.vectors_cache, self.vectors_database, replace_mode=self.replace_mode)
            # still need to keep the difference matrix updated

            self.tp and self.tp.end("addKeyFrame")
            return 'NO LOOP DETECTED!', False

        self.tp and self.tp.end("addKeyFrame")
        return self.__search()

    def getMsgPackage(self):
        """
        Get the message package.
        Returns:
            dict: msg_package
        """
        if self.retrieved_id != 'NO LOOP DETECTED!':
            retrieved_seq_i =\
                self.retrieved_seq_indices[0][0].int().tolist() \
                if "+" in self.backend_name else \
                self.retrieved_seq_indices[0].int().tolist()
            msg_package = {
                'query_timestamp': self.timestamps[-self.matching_ds:],
                'db_timestamp': np.array(self.timestamps)[retrieved_seq_i].tolist(),
                'retrieved_DD': self.retrieved_seq_D,
                'DD': self.DD[:self.search_range].tolist(),
                'querypath': self.querypath,
                'dbpath': self.dbpath,
                'is_pos': str(self.is_positive),
                'retrieved_seq_indices': self.retrieved_seq_indices.tolist(),
                'gt_idx': self.gt_idx,
            }
            if self.backend_name in ["adapt_v4", "adapt_v5", "adapt_v3"]:
                msg_package['saliency_out_idx'] =\
                    self.saliency_block_range.tolist()
                msg_package['Ds'] = self.Ds.tolist()
            if self.DD_enhanced is not None:
                msg_package['DD_enhanced'] = self.DD_enhanced.tolist()
            if self.__isSeqDesc() and self.retrieved_img_indices_seq is not None:
                start_blocks = []
                for retrieved_img_idx in self.retrieved_img_indices_seq:
                    block_start, block_end = retrieved_img_idx - \
                        self.frontend_seqlen + 1, retrieved_img_idx + 1
                    start_blocks.append((block_start, block_end))
                msg_package['start_blocks'] = start_blocks
        else:
            msg_package = {
                'query_timestamp': self.timestamps[-self.matching_ds:],
                'db_timestamp': None,
                'retrieved_DD': None,
                'DD': self.DD[:self.search_range].tolist(),
                'querypath': self.querypath,
                'dbpath': self.dbpath}
        return msg_package

    def isDready(self):
        if self.DD is not None:
            return len(self.DD) > self.stp_gap * self.matching_K + self.matching_ds * 2 + 14
        return False

    def getCurrentGTLst(self, search_idx_limit=None):
        return self.__getGT(-1, search_idx_limit=search_idx_limit)

    def __loadSavedArrayToDeque(self, array):
        if isinstance(array, np.ndarray):
            array_list = [array[i, :] for i in range(array.shape[0])]
            return deque(array_list)
        elif isinstance(array, list):
            return deque(array)
        elif isinstance(array, deque):
            return array
        else:
            raise ValueError("Invalid array type")

    def __loadSavedArrayToList(self, array):
        if isinstance(array, np.ndarray):
            return [array[i, :] for i in range(array.shape[0])]
        elif isinstance(array, list):
            return array
        elif isinstance(array, deque):
            return list(array)
        else:
            raise ValueError("Invalid array type")

    def __appendOrReplace(self, lst, idx, value):
        if idx < len(lst):
            lst[idx] = value
        else:
            lst.append(value)

    def __addOrReplaceCoordIndex(self, id, vector):
        if self.coords_index is None:
            self.__init_coords_index()
        vector_array = np.array([vector], dtype='float32')
        id_array = np.array([id], dtype='int64')
        self.coords_index.remove_ids(id_array)
        self.coords_index.add_with_ids(vector_array, id_array)

    def __isKeyframe(self, current_feature, last_feature, dist_cos_thres):
        dist_cos = 1 - \
            spatial.distance.cosine(np.squeeze(last_feature), np.squeeze(current_feature))

        self.print(f"dist_cos: {dist_cos}")

        return dist_cos <= dist_cos_thres

    def __isSeqDesc(self):
        return "seq_desc" in self.backend_name

    def __init_coords_index(self, add_coords=False):
        self.dim = len(self.coordinates[0])
        index_flat = faiss.IndexFlatL2(self.dim)
        self.coords_index = faiss.IndexIDMap(index_flat)
        if add_coords:
            coords = np.ascontiguousarray(self.coordinates, dtype=np.float32)

            ids = np.ascontiguousarray(
                np.arange(coords.shape[0], dtype=np.int64),
                dtype=np.int64
            )
            self.coords_index.add_with_ids(coords, ids)

    def __getGT(self, idx, search_idx_limit=None):
        # normalize
        if idx < 0:
            idx += len(self.coordinates)

        # grab a view, no copy
        coords = np.array(self.coordinates, dtype=np.float64)
        x0, y0 = coords[idx]

        # compute squared‐distances in one pass
        dx = coords[:, 0] - x0
        dy = coords[:, 1] - y0
        mask = np.sqrt(dx**2 + dy**2) <= self.positive_dist   # boolean array in C

        # extract indices (still numpy)
        neighbors = np.nonzero(mask)[0]

        # drop self
        neighbors = neighbors[neighbors != idx]

        # apply index cap
        if search_idx_limit is not None:
            neighbors = neighbors[neighbors < search_idx_limit]

        return neighbors.tolist()


    def __search(self):

        # Search seq desc
        if self.__isSeqDesc():
            self.tp and self.tp.start("search_seq_desc")
            self.retrieved_id_seq, self.retrieved_img_idx_seq, self.is_positive_seq = self.__searchSeqDesc()
            self.tp and self.tp.end("search_seq_desc")
        self.tp and self.tp.start("search_backend")

        # Search img desc
        if "+" in self.backend_name:
            (self.retrieved_id, self.retrieved_id2), (self.is_positive,
                                                      self.is_positive2) = self.__searchImgDesc()
        else:
            self.retrieved_id, self.is_positive = self.__searchImgDesc()
        self.tp and self.tp.end("search_backend")

        # Set FLAG save_seq
        if self.retrieved_id != 'NO LOOP DETECTED!':
            if self.__isSeqDesc() and not "+" in self.backend_name:
                if self.save_seq_mode == "selected":
                    retrieved_seq_indices = np.array(
                        list(range(self.retrieved_img_idx_seq - self.frontend_seqlen + 1, self.retrieved_img_idx_seq + 1)))
                    is_positive = self.__isPositiveRetrievalSeq(
                        retrieved_seq_indices, search_idx_limit=self.search_range, seq_gt_vgt=self.seq_gt_vgt)
                    self.save_seq = is_positive and not self.is_positive
                else:
                    self.save_seq = True
            elif "+" in self.backend_name:
                self.save_seq = self.is_positive and not self.is_positive2
            else:
                self.save_seq = True

        return self.retrieved_id, self.is_positive

    def __searchSeqDesc(self):
        if self.seq_vectors is not None and not self.fix_search_range and len(self.seq_vectors) > 3 * self.matching_ds:
            # First loop and no fix search range
            # TODO - build global faiss index for 1st loop
            seq_faiss_index = faiss.IndexFlatL2(self.seq_vectors[0].shape[0])
            seq_faiss_index.add(
                np.array(self.seq_vectors[:-(3 * self.matching_ds)]))
            _, pred = seq_faiss_index.search(
                self.seq_vectors[-1][np.newaxis, :], self.matching_K)
            retrieved_img_idx = self.seqID_to_imgID[pred[0, 0]]
            retrieved_id = self.timestamps[retrieved_img_idx]
            if len(self.vectors_database) > (self.matching_ds + self.nosearch_range):
                is_positive = self.__isPositiveRetrieval(
                    retrieved_img_idx, search_idx_limit=self.search_range)
            else:
                is_positive = False
            if self.seq_gt:
                retrieved_seq_indices = np.array(
                    list(range(retrieved_img_idx - self.frontend_seqlen + 1, retrieved_img_idx + 1)))
                is_positive = self.__isPositiveRetrievalSeq(
                    retrieved_seq_indices, search_idx_limit=self.search_range, seq_gt_vgt=self.seq_gt_vgt)

            self.possible_blocks = []
            self.retrieved_img_indices_seq = []
            for p in pred[0]:
                seq_end_idx = self.seqID_to_imgID[p] + self.stp_gap//2 + 1
                seq_start_idx = self.seqID_to_imgID[p] - \
                    self.frontend_seqlen - self.stp_gap//2 + 1
                if seq_start_idx < 0:
                    seq_start_idx = 0
                if seq_end_idx > len(self.vectors_database) - 1:
                    seq_end_idx = len(self.vectors_database) - 1
                self.possible_blocks.append([seq_start_idx, seq_end_idx])
                self.retrieved_img_indices_seq.append(self.seqID_to_imgID[p])
        elif self.fix_search_range:
            # second loop and fix search range in first loop
            if self.seq_index is None:
                # CPU faiss
                # self.seq_index = faiss.IndexFlatL2(self.seq_vectors[0].shape[1])
                # self.seq_index.add(self.seq_vectors[:self.seq_search_range])
                res = faiss.StandardGpuResources()
                self.seq_index = faiss.GpuIndexFlatL2(
                    res, self.seq_vectors[0].shape[0])
                self.seq_index.add(
                    np.array(self.seq_vectors[:self.seq_search_range]))

            _, pred = self.seq_index.search(
                self.seq_vectors[-1][np.newaxis, :], self.matching_K)
            retrieved_img_idx = self.seqID_to_imgID[pred[0, 0]]
            retrieved_id = self.timestamps[retrieved_img_idx]
            is_positive = self.__isPositiveRetrieval(
                retrieved_img_idx, search_idx_limit=self.search_range)
            if self.seq_gt:
                retrieved_seq_indices = np.array(
                    list(range(retrieved_img_idx - self.frontend_seqlen + 1, retrieved_img_idx + 1)))
                is_positive = self.__isPositiveRetrievalSeq(
                    retrieved_seq_indices, search_idx_limit=self.search_range, seq_gt_vgt=self.seq_gt_vgt)

            self.possible_blocks = []
            self.retrieved_img_indices_seq = []
            for p in pred[0]:
                seq_end_idx = self.seqID_to_imgID[p] + self.stp_gap//2 + 1
                seq_start_idx = self.seqID_to_imgID[p] - \
                    self.frontend_seqlen - self.stp_gap//2 + 1
                if seq_start_idx < 0:
                    seq_start_idx = 0
                if seq_start_idx >= self.search_range:
                    seq_start_idx = self.search_range - 1
                if seq_end_idx >= self.search_range:
                    seq_end_idx = self.search_range
                self.possible_blocks.append([seq_start_idx, seq_end_idx])
                self.retrieved_img_indices_seq.append(self.seqID_to_imgID[p])
        else:
            retrieved_id = "NO LOOP DETECTED!"
            retrieved_img_idx = None
            is_positive = False
        return retrieved_id, retrieved_img_idx, is_positive

    def __searchImgDesc(self):
        if self.D is None:
            self.D = self.__getDifferenceMatrix(
                self.vectors_cache, self.vectors_database)
        else:
            self.D = self.__updateDifferenceMatrix(
                self.vectors_cache, self.vectors_database, replace_mode=self.replace_mode)
        self.DD = self.D

        # Difference matrix is smaller than minimal required size, skip
        if not self.isDready():
            retrieved_id, is_positive = 'NO LOOP DETECTED!', False
            return retrieved_id, is_positive

        if "+" in self.backend_name:
            retrieved_id, is_positive = self.__searchImgDualBackend()
        elif self.backend_name == "none":
            retrieved_id, is_positive = self.__searchImgSingle()
        # ANCHOR Normal mode
        else:
            retrieved_id, is_positive = self.__searchImgSeq()
        return retrieved_id, is_positive

    def __searchImgSeq(self):
        self.retrieved_seq_indices, tf_value, self.retrieved_seq_D = self.__getMatches(
            self.DD, search_idx_limit=self.search_range, backend_name=self.backend_name)
        if self.retrieved_seq_indices is not None:
            if self.global_min_value > tf_value:
                self.global_min_value = float(tf_value)
            self.print(
                f"Retrieved idx: {int(self.retrieved_seq_indices[0][-1])}, tf_value: {tf_value}")
            self.print(f'Global tf value: {self.global_min_value}')

            is_positive = self.__isPositiveRetrieval(
                self.retrieved_seq_indices[0][-1].int(), search_idx_limit=self.search_range)
            retrieved_id = self.timestamps[self.retrieved_seq_indices[0][-1].int(
            )] if tf_value < self.thresh_out else 'NO LOOP DETECTED!'
            if self.seq_gt:
                is_positive = self.__isPositiveRetrievalSeq(
                    self.retrieved_seq_indices[0].numpy(), search_idx_limit=self.search_range, seq_gt_vgt=self.seq_gt_vgt)

            return retrieved_id, is_positive
        else:
            return 'NO LOOP DETECTED!', False

    # ANCHOR Single image mode

    def __searchImgSingle(self):
        """
        Search in single image mode, recallN would be saved in self.is_positive_N

        """
        self.retrieved_seq_indices, tf_value, retrieved_seq_D = self.__getMatches(
            self.DD, search_idx_limit=self.search_range, backend_name="seqslam")
        self.is_positive_1.append(self.__isPositiveRetrieval(
            self.retrieved_seq_indices.squeeze().numpy()[0], search_idx_limit=self.search_range))
        self.is_positive_5.append(np.isin(
            self.retrieved_seq_indices.squeeze().numpy()[:5], self.gt_idx[-1]).any())
        self.is_positive_10.append(np.isin(
            self.retrieved_seq_indices.squeeze().numpy()[:10], self.gt_idx[-1]).any())
        self.is_positive_20.append(np.isin(
            self.retrieved_seq_indices.squeeze().numpy()[:20], self.gt_idx[-1]).any())

        is_positive = self.is_positive_1[-1]
        retrieved_id = self.timestamps[self.retrieved_seq_indices[0][-1].int(
        )] if tf_value < self.thresh_out else 'NO LOOP DETECTED!'

        return retrieved_id, is_positive

    # ANCHOR Dual backend mode
    def __searchImgDualBackend(self):
        """
            Search for the backend in dual backend mode. The first backend's result is used to determined if a loop is detected.
        """
        backend1, backend2 = self.backend_name.split("+")
        retrieved_seq_indices1, tf_value, retrieved_seq_D1 = self.__getMatches(
            self.DD, search_idx_limit=self.search_range, backend_name=backend1)
        retrieved_seq_indices2, tf_value2, retrieved_seq_D2 = self.__getMatches(
            self.DD, search_idx_limit=self.search_range, backend_name=backend2)
        if retrieved_seq_indices1 is not None and retrieved_seq_indices2 is not None:
            self.retrieved_seq_indices = torch.cat(
                (retrieved_seq_indices1.unsqueeze(0), retrieved_seq_indices2.unsqueeze(0)), dim=0)
            self.retrieved_seq_D = [retrieved_seq_D1, retrieved_seq_D2]

            is_positive = self.__isPositiveRetrieval(
                retrieved_seq_indices1[0][-1].int(), search_idx_limit=self.search_range)

            is_positive2 = retrieved_seq_indices2[0][-1] in self.gt_idx[-1]

            if self.seq_gt:
                is_positive = self.__isPositiveRetrievalSeq(
                    retrieved_seq_indices1[0].numpy(), search_idx_limit=self.search_range, seq_gt_vgt=self.seq_gt_vgt)
                is_positive2 = self.__isPositiveRetrievalSeq(
                    retrieved_seq_indices2[0].numpy(), search_idx_limit=self.search_range, seq_gt_vgt=self.seq_gt_vgt)

            retrieved_id = self.timestamps[self.retrieved_seq_indices[0][0][-1].int(
            )] if tf_value < self.thresh_out else 'NO LOOP DETECTED!'
            retrieved_id2 = self.timestamps[self.retrieved_seq_indices[1][0]
                                            [-1].int()] if tf_value2 < self.thresh_out2 else 'NO LOOP DETECTED!'

            return (retrieved_id, retrieved_id2), (is_positive, is_positive2)
        else:
            return ('NO LOOP DETECTED!', 'NO LOOP DETECTED!'), (False, False)

    def __getDifferenceMatrix(self, queries, database):
        if not isinstance(queries, np.ndarray):
            queries = np.array(queries)
        if not isinstance(database, np.ndarray):
            database = np.array(database)

        # queries, database => numpy.array
        # shape = (num of v, dim of v)
        qFeats = torch.from_numpy(queries.copy()).float()
        dbFeats = torch.from_numpy(database.copy()).float()

        if self.sim == "l2":
            # Calculating L2 distance
            # Expand dimensions to broadcast and calculate pairwise distances
            qFeats = qFeats.unsqueeze(0)  # Shape: [num_queries, 1, dim]
            dbFeats = dbFeats.unsqueeze(1)  # Shape: [1, num_database, dim]
            D = torch.sqrt(torch.sum((dbFeats - qFeats) ** 2, dim=2))/2
        else:
            # Default to cosine similarity
            # Use your original method for cosine similarity
            D = 1 - torch.matmul(dbFeats, qFeats.T) / torch.matmul(torch.norm(
                dbFeats, p=2, dim=1).unsqueeze(-1), torch.norm(qFeats, p=2, dim=1).unsqueeze(0))

        return D

    def __updateDifferenceMatrix(self, queries, database, replace_mode=False):
        no_row_added = self.fix_search_range and len(
            self.D) == self.search_range
        if self.fix_search_range:
            assert len(
                self.D) <= self.search_range, "Search range is fixed, but D is larger than search range"

        # Add new comlumn to the difference matrix
        if no_row_added:

            if self.dbFeats_fix is None:
                if not isinstance(database, np.ndarray):
                    dbFeats_np = np.array(database[:self.search_range])
                else:
                    dbFeats_np = database[:self.search_range]
                self.dbFeats_fix = torch.from_numpy(
                    dbFeats_np).float().to(self.device)
            dbFeats = self.dbFeats_fix
        elif replace_mode:
            if not isinstance(database, np.ndarray):
                dbFeats_np = np.array(database)
            else:
                dbFeats_np = database
            dbFeats = torch.from_numpy(dbFeats_np).float().to(self.device)
        else:
            if not isinstance(database, np.ndarray):
                dbFeats_np = np.array(database[:-1])
            else:
                dbFeats_np = database[:-1]

            dbFeats = torch.from_numpy(dbFeats_np).float().to(self.device)
        if not isinstance(queries, np.ndarray):
            queries = np.array(queries)
        qFeat = torch.from_numpy(
            queries[-1]).reshape(1, -1).float().to(self.device)

        if self.sim == "l2":
            # Calculate L2 distance for the new column
            new_column = torch.sqrt(
                torch.sum((dbFeats - qFeat) ** 2, dim=1)).unsqueeze(1)/2
        else:
            # Calculate cosine similarity for the new column
            new_column = 1 - torch.matmul(dbFeats, qFeat.T) / torch.matmul(torch.norm(
                dbFeats, p=2, dim=1).unsqueeze(-1), torch.norm(qFeat, p=2, dim=1).unsqueeze(0))
        new_column = new_column.cpu()

        if replace_mode:
            # In replace_mode, replace the last column
            D = torch.cat((self.D[:, :-1], new_column), 1)
        else:
            # In normal mode, append the new column after removing the first one
            D = torch.cat((self.D[:, 1:], new_column), 1)

        # Add new row to the difference matrix
        if not no_row_added and not replace_mode:
            qFeats = torch.from_numpy(queries).float()
            dbFeat = torch.from_numpy(database[-1]).reshape(1, -1).float()
            if self.sim == "l2":
                # Calculate L2 distance for the new row
                new_row = torch.sqrt(
                    torch.sum((dbFeat - qFeats) ** 2, dim=1)).unsqueeze(0)/2
            else:
                # Calculate cosine similarity for the new row
                new_row = 1 - torch.matmul(dbFeat, qFeats.T) / torch.matmul(torch.norm(
                    dbFeat, p=2, dim=1).unsqueeze(-1), torch.norm(qFeats, p=2, dim=1).unsqueeze(0))
                new_row = new_row/2

            # Update the matrix D with the new row
            D = torch.cat((D, new_row), 0)

        return D

    def __isPositiveRetrieval(self, idx, search_idx_limit=None):
        if search_idx_limit is None:
            search_idx_limit = -(self.matching_ds + self.nosearch_range)
        if self.gt_idx == self.matching_ds:
            gt_lst = self.gt_idx[-1]
        else:
            gt_lst = self.getCurrentGTLst(search_idx_limit)
        return np.isin(idx, gt_lst).any()

    def __updateGTIdx(self):
        if self.search_range is None:
            search_idx_limit = -(self.matching_ds + self.nosearch_range)
        else:
            search_idx_limit = self.search_range
        gt_lst = self.getCurrentGTLst(search_idx_limit)
        if len(self.gt_idx) < self.matching_ds:
            for i in range(self.matching_ds, 0, -1):
                gt_lst = self.__getGT(-i, search_idx_limit=search_idx_limit)
                self.gt_idx.append(gt_lst)
        else:
            self.gt_idx.pop(0)
            self.gt_idx.append(gt_lst)

    def __isPositiveRetrievalSeq(self, retrieved_seq, search_idx_limit=None, seq_gt_vgt=False):
        if search_idx_limit is None:
            search_idx_limit = -(self.matching_ds + self.nosearch_range)

        gt_lst = []
        for i in range(-1, -self.matching_ds-1, -1):
            gt_lst += self.__getGT(i, search_idx_limit=search_idx_limit)
        gt_lst = list(set(gt_lst))
        self.seqgt_lst = gt_lst

        if seq_gt_vgt:
            return np.isin(retrieved_seq, gt_lst).any()
        else:
            return np.isin(retrieved_seq, gt_lst).all()

    def __correctVectorsCache(self):
        if len(self.vectors_cache) > self.matching_ds:
            for _ in range(len(self.vectors_cache) - self.matching_ds):
                self.vectors_database.append(self.vectors_cache.popleft())
        elif len(self.vectors_cache) < self.matching_ds:
            for _ in range(self.matching_ds - len(self.vectors_cache)):
                self.vectors_cache.appendleft(self.vectors_database.pop())

    def __getMatches(self, DD, search_idx_limit=None, backend_name="seqslam"):
        if search_idx_limit is None:
            search_idx_limit = len(DD)-self.nosearch_range
        elif backend_name == "seqslam":
            return self.__seqslam(DD, search_idx_limit)
        elif backend_name == "seq_desc_seqmatch":
            return self.__seqmatch(DD, self.retrieved_img_indices_seq, search_idx_limit)
        elif backend_name == "adaptseq":
            return self.__adaptseq(DD, search_idx_limit)
        elif backend_name == "seq_desc":
            return self.__seq_desc(DD)
        elif backend_name == "seq_desc_adaptseq_v2":
            return self.__seq_desc_adaptseq_v2(DD, self.retrieved_img_indices_seq, search_idx_limit)

    def __seqslam(self, DD, search_idx_limit=None):
        DD = DD[:search_idx_limit]
        self.print(f"Max of DD: {DD.max()}, Min of DD: {DD.min()}")
        # Under Construction
        matches = np.nan*np.ones((DD.shape[1], 2))
        # for N in range(self.matching_ds//2, DD.shape[1]-self.matching_ds//2):
        move_min = self.matching_vmin * self.matching_ds
        move_max = self.matching_vmax * self.matching_ds
        move = torch.arange(int(move_min), int(move_max)+1)
        v = move.float() / self.matching_ds
        idx_add = torch.arange(0, self.matching_ds).repeat((len(v), 1))
        idx_add = torch.floor(idx_add * v.repeat((idx_add.shape[1], 1)).T)
        x = torch.arange(1, self.matching_ds + 1).repeat(len(v), 1)
        y_max = DD.shape[0]
        flatDD = DD.T.flatten()
        # only search the top N most familiar templates
        search_range = DD[:, 0].argsort()[:self.matching_K]
        score = torch.zeros(search_range.shape[0])
        matched_addidx = torch.zeros(
            search_range.shape[0], idx_add.shape[1])

        for i, s in enumerate(search_range):
            y = torch.clone(idx_add + s + 1)
            y[y > y_max] = y_max
            idx = ((x-1) * y_max + y).long()
            ds = torch.sum(flatDD[idx - 1], 1)
            optimal_move = torch.argmin(ds)
            score[i] = ds[optimal_move]
            matched_addidx[i] = idx_add[optimal_move, :]

        min_idx = search_range[torch.argmin(score)]
        # min_value=score[min_idx]
        min_value = torch.min(score)
        min_value_2nd = score.sort().values[1]

        out_idx = search_range[score.argsort()].unsqueeze(-1) + \
            matched_addidx[score.argsort()]

        out_idx[out_idx > DD.shape[0] - 1] = DD.shape[0] - 1
        out_idx = out_idx.int()

        retrieved_DD = [float(DD[out_idx[0][i].int(), i])
                        for i in range(self.matching_ds)]

        value = min_value / min_value_2nd

        self.posP = value
        return out_idx, value, retrieved_DD

    def __seqmatch(self, DD, retrieved_img_indices_seq,   search_idx_limit=None):
        """
        Changes to v2:
            Use sequence result to find starting point.

        """
        DD = DD[:search_idx_limit]
        self.print(f"Max of DD: {DD.max()}, Min of DD: {DD.min()}")
        # for N in range(self.matching_ds//2, DD.shape[1]-self.matching_ds//2):
        v = torch.tensor([1]).float()
        idx_add = torch.arange(0, self.matching_ds).repeat((len(v), 1))
        idx_add = torch.floor(idx_add * v.repeat((idx_add.shape[1], 1)).T)
        x = torch.arange(1, self.matching_ds + 1).repeat(len(v), 1)
        y_max = DD.shape[0]
        flatDD = DD.T.flatten()
        if retrieved_img_indices_seq is None:
            # only search the top N most familiar templates
            search_range = DD[:, 0].argsort()[:self.matching_K]
        else:
            search_range = []
            for retrieved_img_idx in retrieved_img_indices_seq:
                start = retrieved_img_idx - self.frontend_seqlen + 1
                search_range.append(start)
            search_range = torch.tensor(search_range)
        score = torch.zeros(search_range.shape[0])
        matched_addidx = torch.zeros(
            search_range.shape[0], idx_add.shape[1])

        for i, s in enumerate(search_range):
            y = torch.clone(idx_add + s + 1)
            y[y > y_max] = y_max
            idx = ((x-1) * y_max + y).long()
            ds = torch.sum(flatDD[idx - 1], 1)
            optimal_move = torch.argmin(ds)
            score[i] = ds[optimal_move]
            # matched_addidx[i] = int(idx_add[optimal_move,-1])
            matched_addidx[i] = idx_add[optimal_move, :]

        min_idx = search_range[torch.argmin(score)]
        min_value = torch.min(score)
        min_value_2nd = score.sort().values[1]

        out_idx = search_range[score.argsort()].unsqueeze(-1) + \
            matched_addidx[score.argsort()]

        out_idx[out_idx > DD.shape[0] - 1] = DD.shape[0] - 1
        out_idx = out_idx.int()

        retrieved_DD = [float(DD[out_idx[0][i].int(), i])
                        for i in range(self.matching_ds)]

        value = min_value / min_value_2nd

        self.posP = value
        return out_idx, value, retrieved_DD

    def __adaptseq(self, DD, search_idx_limit=None):
        """
            Starting points: 
                sumpool (stride is "stp_gap") to pick top "matching_K" possible block, then select min of each block.
            Search: 
                dual-direction adaptive search
            Score: 
                all value along traj added together
        """
        sumpool = torch.nn.AvgPool2d(
            (self.stp_gap, self.matching_ds), stride=self.stp_gap)
        seletected_b_idx = sumpool(DD[:search_idx_limit, :].unsqueeze(
            0).unsqueeze(0)).squeeze().argsort()[:self.matching_K]
        search_ridx_lst = []
        search_cidx_lst = []
        for b_idx in seletected_b_idx.tolist():
            b_start_idx = b_idx * self.stp_gap
            b_end_idx = b_start_idx + self.stp_gap
            if b_end_idx > search_idx_limit:
                b_end_idx = search_idx_limit
            c_min_values, c_mins_cidx = DD[b_start_idx:b_end_idx, :].min(1)
            c_min_ridx = c_min_values.argsort()[0]
            c_min_cidx = c_mins_cidx[c_min_ridx]
            search_ridx_lst.append(c_min_ridx + b_start_idx)
            search_cidx_lst.append(c_min_cidx)
        search_cidx_lst = torch.tensor(search_cidx_lst)
        search_ridx_lst = torch.tensor(search_ridx_lst)

        idx = torch.zeros(self.matching_K, self.matching_ds, dtype=int)
        idx[range(self.matching_K), search_cidx_lst] = search_ridx_lst
        score = DD[search_ridx_lst, search_cidx_lst]

        for i in range(search_ridx_lst.shape[0]):
            for j in range(search_cidx_lst[i], 0, -1):
                start = idx[i, j] + 1 - self.matching_KK
                if start < 0:
                    idx[i, j - 1] = DD[0:start +
                                       self.matching_KK, j - 1].argmin()
                else:
                    idx[i, j - 1] = DD[start:start +
                                       self.matching_KK, j-1].argmin() + start
                score[i] += DD[idx[i, j-1], j-1]
            for j in range(search_cidx_lst[i], self.matching_ds-1, 1):
                start = idx[i, j]
                if start + self.matching_KK > search_idx_limit:
                    idx[i, j + 1] = DD[start:search_idx_limit, j + 1].argmin() + \
                        start
                else:
                    idx[i, j + 1] = DD[start:start +
                                       self.matching_KK, j + 1].argmin() + start
                score[i] += DD[idx[i, j+1], j+1]

        retrieved_idx = idx[torch.argmin(score), -1]
        out_idx = idx[score.argsort()]
        min_value = score.min()
        retrieved_DD = [float(DD[out_idx[0][i].int(), i])
                        for i in range(self.matching_ds)]
        return out_idx, min_value, retrieved_DD

    def __seq_desc(self, DD):
        '''
            Use the last frame as retrieved frame.
            64.05
        '''
        if self.retrieved_img_idx_seq is not None:
            out_idx = torch.full(
                (self.matching_K, self.matching_ds), self.retrieved_img_idx_seq, dtype=int)
            min_value = 0  # As per the requirement
            retrieved_DD = [float(DD[out_idx[0][i].int(), i])
                            for i in range(self.matching_ds)]
        else:
            out_idx = None
            min_value = 999
            retrieved_DD = None
        return out_idx, min_value, retrieved_DD

    def __seq_desc_adaptseq_v2(self, DD, retrieved_img_indices_seq, search_idx_limit=None):
        """
            Modified from v16_noenhance, no norm for D
        """

        self.print(f"Max of DD: {DD.max()}, Min of DD: {DD.min()}")

        search_ridx_lst = []
        search_cidx_lst = []

        if retrieved_img_indices_seq is None:
            r_mins_values, r_mins_cidx = DD[:search_idx_limit, :].min(1)
            r_mins_ridx = r_mins_values.argsort()
            search_ridx_lst = (r_mins_ridx[0]).view(1)
            r_mins_ridx = r_mins_ridx[1:]
            for _ in range(1, self.matching_K):
                while torch.any(abs(r_mins_ridx[0] - search_ridx_lst) < self.stp_gap):
                    r_mins_ridx = r_mins_ridx[1:]
                search_ridx_lst = torch.cat(
                    (search_ridx_lst, r_mins_ridx[0].view(1)), 0)
                r_mins_ridx = r_mins_ridx[1:]
            search_cidx_lst = r_mins_cidx[search_ridx_lst]
        else:
            for retrieved_img_idx in retrieved_img_indices_seq:
                block_start, block_end = retrieved_img_idx - \
                    self.frontend_seqlen + 1, retrieved_img_idx + 1
                DD_block = DD[block_start:block_end, :]
                r_mins_value, r_mins_cidx = DD_block[:, -
                                                     self.frontend_seqlen:].min(1)
                r_min_ridx = r_mins_value.argsort()[0]
                r_min_cidx = r_mins_cidx[r_min_ridx]

                # Mapping to original DD coordinates
                search_ridx_lst.append(r_min_ridx + block_start)
                search_cidx_lst.append(
                    self.matching_ds - self.frontend_seqlen + r_min_cidx)

            search_cidx_lst = torch.tensor(search_cidx_lst)
            search_ridx_lst = torch.tensor(search_ridx_lst)

        DD_enhanced = DD[:search_idx_limit]

        idx = torch.zeros(self.matching_K, self.matching_ds, dtype=int)
        idx[range(self.matching_K), search_cidx_lst] = search_ridx_lst
        score = DD_enhanced[search_ridx_lst, search_cidx_lst]

        for i in range(search_ridx_lst.shape[0]):
            for j in range(search_cidx_lst[i], 0, -1):
                start = idx[i, j] + 1 - self.matching_KK
                if start < 0:
                    idx[i, j - 1] = DD_enhanced[0:start +
                                                self.matching_KK, j - 1].argmin()
                else:
                    idx[i, j - 1] = DD_enhanced[start:start +
                                                self.matching_KK, j-1].argmin() + start
                score[i] += DD_enhanced[idx[i, j-1], j-1]
            for j in range(search_cidx_lst[i], self.matching_ds-1, 1):
                start = idx[i, j]
                if start + self.matching_KK > search_idx_limit:
                    idx[i, j + 1] = DD_enhanced[start:search_idx_limit, j + 1].argmin() + \
                        start
                else:
                    idx[i, j + 1] = DD_enhanced[start:start +
                                                self.matching_KK, j + 1].argmin() + start
                score[i] += DD_enhanced[idx[i, j+1], j+1]

        retrieved_idx = idx[torch.argmin(score), -1]
        self.rerank_original_idx = score.argsort()
        self.score = score
        out_idx = idx[score.argsort()]
        min_value = score.min()
        retrieved_DD = [float(DD[out_idx[0][i].int(), i])
                        for i in range(self.matching_ds)]
        self.DD_enhanced = DD_enhanced
        return out_idx, min_value, retrieved_DD
