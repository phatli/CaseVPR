import os, sys
import pickle
import cv2
import numpy as np
import torch
import inspect
import json
from datetime import datetime
from os import makedirs
from os.path import exists

from .utils import CASEVPR_ROOT_DIR
if CASEVPR_ROOT_DIR not in sys.path:
    sys.path.insert(0, CASEVPR_ROOT_DIR)

from scripts.configs.model_configs import img_encoders, seq_encoders, hvpr_encoders
from .frontend import seqFrontEnd
from .backend import seqBackEnd, seqbackend_params
from .utils import TimeProbe, vis_seq


def get_analysis_result(nTP, nFP, nTN, nFN):
    prec_denom = float(nTP+nFP)
    accu_denom = float(nTP+nFN+nFP+nTN)
    recall_denom = float(nTP+nFN)
    precision = nTP/prec_denom if prec_denom else 0
    accuracy = (nTP+nTN)/accu_denom if accu_denom else 0
    recall = nTP/recall_denom if recall_denom else 0
    analysis_result = {
        'precision': precision,
        'accuracy': accuracy,
        "recall": recall,
        'nTP': nTP,
        'nFP': nFP,
        'nTN': nTN,
        'nFN': nFN,
    }

    return analysis_result


def get_testName(dataset_name, dataset2_name, from_scratch, model_name, test_name, seqbackend_params):
    return f"{datetime.now().strftime('%Y%m%d%H%M')}-{dataset_name}-{dataset2_name}-{model_name}-{seqbackend_params['backend_name']}-{seqbackend_params.get('pre_name','')}-{1 if from_scratch else 2}-{test_name}"

def wrapper_print(*text, color="white"):
    # print(*text)
    pass

class LoopDetector():
    """Loop Detector class
    This class is used to detect loops in a sequence of images using a sequence encoder and a sequence backend.
    It is designed to be used with the CASEVPR framework.
    Args:
        dataset_name (str): name of the dataset to be used for testing.
        dataset2_name (str): name of the second dataset to be used for testing.
        from_scratch (bool): whether to train the model from scratch or not.
        model_name (str): name of the model to be used for testing.
        test_name (str): name of the test to be used for testing.
        seqbackend_params (dict): parameters for the sequence backend.
        save_seq (bool): whether to save the sequence or not.
        save_retrieved_info (bool): whether to save the retrieved information or not.
        seperate_ds (bool): whether to use separate datasets or not.
        debug (bool): whether to use debug mode or not.
        save_retrieval (bool): whether to save the retrieval or not.
        save_feature_cache (bool): whether to save the feature cache or not.
        save_result_every_frame (bool): whether to save the result every frame or not.
        showConsole (bool): whether to show console output or not.
        only_log_vectors (bool): whether to only log vectors or not.
        img_encoders (dict): image encoders settings.
        seq_encoders (dict): sequence encoders settings.
        hvpr_encoders (dict): hvpr encoders settings.
    """
    def __init__(self, dataset_name, dataset2_name, from_scratch, model_name="netvlad_WPCA4096", test_name="", seqbackend_params=seqbackend_params, save_seq=False, save_retrieved_info=False, seperate_ds=False, debug=False, save_retrieval=False, save_feature_cache=True, save_result_every_frame=True, showConsole=True, only_log_vectors=False):
        self.img_idx = 0
        self.save_retrieved_info = not debug and not from_scratch and save_retrieved_info
        self.save_retrieval = not debug and not from_scratch and save_retrieval
        self.save_feature_cache = not debug and not from_scratch and save_feature_cache
        self.save_result_every_frame = not debug and save_result_every_frame # Save result every frame, otherwise only save at the end
        self.tp = TimeProbe(20) if showConsole else None
        self.print = self.tp.print if showConsole else wrapper_print
        encoder_configs = (img_encoders, seq_encoders, hvpr_encoders)

        self.raw_dataset_name = dataset_name
        self.raw_dataset2_name = dataset2_name
        safe_dataset_name = dataset_name.replace("/", "_")
        safe_dataset2_name = dataset2_name.replace("/", "_")

        self.seq_encoder = seqFrontEnd(model_name, encoder_configs, tp = self.tp, print_func=self.print)
        self.seq_backend = seqBackEnd(
            frontend_name=model_name,
            frontend_seqlen=self.seq_encoder.seqlen,
            dataset_name=safe_dataset_name,
            dataset2_name=safe_dataset2_name,
            **seqbackend_params,
            print_func=self.print, from_scratch=from_scratch, tp=self.tp, only_log_vectors=only_log_vectors)

        self.save_search_status = not seqbackend_params["backend_name"] in [
            "none"]
        self.__init_stat()
        self.from_scratch = from_scratch
        self.testName = get_testName(
            safe_dataset_name, safe_dataset2_name, from_scratch, model_name, test_name, seqbackend_params)
        self.dataset_name = safe_dataset_name if from_scratch else safe_dataset2_name
        self.debug = debug

        self.result_save_dir = os.path.join(
            CASEVPR_ROOT_DIR, "output", "test_logs", self.testName)
        self.clock = 0
        if not exists(self.result_save_dir) and not self.debug:
            makedirs(self.result_save_dir)

        self.save_seq = save_seq
        self.seperate_ds = (not from_scratch) and seperate_ds
        self.do_search = True  # Flag shows whether to search or not
        if self.seperate_ds:
            self.do_search = False

        if self.save_seq:
            self.saveSeq_dir = os.path.join(
                CASEVPR_ROOT_DIR, f"output/test_logs/{self.testName}", "seq")
            if not exists(self.saveSeq_dir):
                makedirs(self.saveSeq_dir)

        self.feature_name = f"{safe_dataset2_name}-{model_name}" if not self.from_scratch else f"{safe_dataset_name}-{model_name}"
        self.feature_path = os.path.join(
            CASEVPR_ROOT_DIR, "output", "feature_cache", self.feature_name)

        if not self.from_scratch:
            _ = self.seq_backend.loadCheckpoint(freeze_search_range=True, seperate_ds=seperate_ds)

        if self.save_retrieved_info:
            self.r_idx = 0
        if self.save_retrieval:
            self.r_idx = 0

        if not self.debug:
            arg_dict = {k: v for k, v in locals().items(
            ) if k in inspect.signature(self.__init__).parameters}
            self.__save_config(arg_dict)

    def __init_stat(self):
        self.nTP = 0
        self.nFP = 0
        self.nTN = 0
        self.nFN = 0
        if "+" in self.seq_backend.backend_name:
            self.nTP2 = 0
            self.nFP2 = 0
            self.nTN2 = 0
            self.nFN2 = 0

    def __callback(self, img, pose, img_vector_cache=None, seq_vector_cache=None):
        self.clock += 1
        self.print(f"Processing {self.clock}th image...")
        self.tp and self.tp.start("callback")
        image_p = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# region Change flag to search
        if self.seperate_ds and self.seq_backend.local_idx == self.seq_backend.matching_ds - 1:
            self.seq_backend.search_range = len(
                self.seq_backend.vectors_database) + 1
            self.seq_backend.fix_search_range = True
            self.do_search = True
# endregion

# region Get vectors and search

        img_vector, seq_vector = None, None
        if "seq_desc" in self.seq_backend.backend_name:
            self.tp and self.tp.start("get_vector")
            torch.cuda.synchronize()
            if self.seq_backend.stacked_images is not None and len(self.seq_backend.stacked_images) == self.seq_backend.frontend_seqlen - 1:
                img_p = image_p[np.newaxis, :]
                
                if img_vector_cache is not None and seq_vector_cache is not None:
                    img_vector, seq_vector = img_vector_cache, seq_vector_cache
                else:
                    seq = np.concatenate(
                    (np.array(self.seq_backend.stacked_images), img_p), axis=0)
                    _, img_vector, seq_vector = self.seq_encoder.get_vector(
                        image_p, seq=seq)
            else:
                if img_vector_cache is not None:
                    img_vector = img_vector_cache
                else:
                    _, img_vector, _ = self.seq_encoder.get_vector(image_p)

            torch.cuda.synchronize()
            self.tp and self.tp.end("get_vector")

            retrieved_id, is_positive = self.seq_backend.addKeyFrame(
                str(self.clock), img_vector, img, (pose[0], pose[1]), seq_feature=seq_vector, no_search=not self.do_search)
        else:
            self.tp and self.tp.start("get_vector")
            if img_vector_cache is not None:
                img_vector = img_vector_cache
            else:
                _, img_vector, _ = self.seq_encoder.get_vector(image_p)
            self.tp and self.tp.end("get_vector")

            retrieved_id, is_positive = self.seq_backend.addKeyFrame(
                str(self.clock), img_vector, img, (pose[0], pose[1]), no_search=not self.do_search)


# endregion

# region Save feature cache
        if self.save_feature_cache and img_vector_cache is None and seq_vector_cache is None:
            vectors = {
                "img_vector": img_vector,
                "seq_vector": seq_vector
            }
            if not exists(self.feature_path):
                makedirs(self.feature_path)
            with open(os.path.join(self.feature_path, f"{self.clock - 1:04d}.pkl"), "wb") as f:
                pickle.dump(vectors, f)

# endregion

# region Print and save results
        self.tp and self.tp.start("result analysis")
        if retrieved_id != 'NOT A KEY FRAME!' and len(self.seq_backend.coordinates) >= 2 and self.do_search:
            if retrieved_id != 'NO LOOP DETECTED!':
                if is_positive:
                    self.nTP += 1
                elif not len(self.seq_backend.getCurrentGTLst(self.seq_backend.search_range)) == 0:
                        self.nFP += 1
            else:
                if is_positive:
                    self.nFN += 1
                else:
                    self.nTN += 1

            self.print(
                f"{self.seq_backend.backend_name.split('+')[0]} (thresh:{self.seq_backend.thresh_out}):", color="red")
            self.print(
                f" {'correct' if is_positive else 'incorrect'} loop detected!,nTP:{self.nTP},nFP:{self.nFP},nTN:{self.nTN},nFN:{self.nFN}, total:{self.nTP+self.nFP+self.nTN+self.nFN}"
            )

            if "+" in self.seq_backend.backend_name:
                if self.seq_backend.retrieved_id2 != 'NO LOOP DETECTED!':
                    if self.seq_backend.is_positive2:
                        self.nTP2 += 1
                    elif self.seq_backend.is_positive2 == False and not len(self.seq_backend.getCurrentGTLst()) == 0:
                        self.nFP2 += 1
                else:
                    if self.seq_backend.is_positive2:
                        self.nFN2 += 1
                    else:
                        self.nTN2 += 1

                self.print(
                    f"{self.seq_backend.backend_name.split('+')[1]} (thresh:{self.seq_backend.thresh_out2}):", color="red")
                self.print(
                    f" {'correct' if self.seq_backend.is_positive2 else 'incorrect'} loop detected!,nTP:{self.nTP2},nFP:{self.nFP2},nTN:{self.nTN2},nFN:{self.nFN2}"
                )

            # ANCHOR Save Sequence Visualization
            if self.save_search_status and self.seq_backend.isDready() and self.save_seq:
                msg_package = self.seq_backend.getMsgPackage()
                vis_seq(msg_package, self.img_idx,
                        self.seq_backend.backend_name, self.seq_backend.save_seq, self.saveSeq_dir)
            self.img_idx += 1

            # ANCHOR Save analysis results
            if self.save_result_every_frame:
                self.saveResult()

            # ANCHOR Save retrieved info of each frame, including retrieved sequence's frame indices and ground truth indices
            if self.save_retrieved_info and self.seq_backend.posP is not None:
                retrieved_dir = os.path.join(
                    self.result_save_dir, 'retrieved_info')
                if not exists(retrieved_dir):
                    makedirs(retrieved_dir)

                r_save = {
                    'is_pos': str(is_positive),
                    'posP': self.seq_backend.posP,
                    'retrieved_seq_indices': self.seq_backend.retrieved_seq_indices,
                    'gt_idx': self.seq_backend.gt_idx,
                }

                with open(retrieved_dir + f"/{self.r_idx:04d}.pkl", "wb") as f:
                    pickle.dump(r_save, f)
                self.r_idx += 1

            # * ANCHOR Save retrieved sequence's frame indices and coordinates of query and database
            if self.save_retrieval:
                retrieved_dir = os.path.join(self.result_save_dir, 'retrieval')
                if not exists(retrieved_dir):
                    makedirs(retrieved_dir)
                database_coords_path = os.path.join(
                    self.result_save_dir, 'database_coords.pkl')
                if not exists(database_coords_path):
                    with open(database_coords_path, "wb") as f:
                        pickle.dump(
                            self.seq_backend.coordinates[:self.seq_backend.search_range], f)
                r_save = {
                    "retrieved_seq_indices": self.seq_backend.retrieved_seq_indices,
                    "coord": self.seq_backend.coordinates[-1]
                }
                with open(os.path.join(retrieved_dir, f"{self.r_idx:04d}.pkl"), "wb") as f:
                    pickle.dump(r_save, f)
                self.r_idx += 1

        self.tp and self.tp.end("result analysis")
# endregion

        if self.tp is not None:
            self.tp.end("callback")
            self.tp.print(self.tp.getResults(), color="green")
            self.tp.refresh()
        return retrieved_id, is_positive

    def saveResult(self):
        analysis_result = get_analysis_result(
            self.nTP, self.nFP, self.nTN, self.nFN)

        if "+" in self.seq_backend.backend_name:
            analysis_result2 = get_analysis_result(
                self.nTP2, self.nFP2, self.nTN2, self.nFN2)
            be1, be2 = self.seq_backend.backend_name.split('+')
            analysis_result = {be1: analysis_result, be2: analysis_result2}

        with open(os.path.join(self.result_save_dir, 'result.json'), 'w') as f:
            json.dump(analysis_result, f, indent=4)

    def __save_config(self, arg_dict):
        with open(os.path.join(self.result_save_dir, 'config.json'), 'w') as f:
            json.dump(arg_dict, f, indent=4)

    def callback(self, img, pose, use_cache=True):
        feature_cache_file = os.path.join(
            self.feature_path, f"{self.clock:04d}.pkl")
        error_flag = False
        if use_cache and exists(feature_cache_file):
            self.print(f"Using features cache...")
            try:
                with open(feature_cache_file, "rb") as f:
                    vectors = pickle.load(f)
            except Exception as e:  # Catching any exception related to file I/O or unpickling
                self.print(
                    f"Failed to load data from cache due to error: {str(e)}. The file may be corrupted, incomplete, or other issues.")
                error_flag = True
            return self.__callback(img, pose) if error_flag else self.__callback(
                img, pose, img_vector_cache=vectors["img_vector"], seq_vector_cache=vectors["seq_vector"])
        else:
            self.print(f"Extracting features...")
            return self.__callback(img, pose)

    def saveSeqCheckpoint(self):
        self.seq_backend.saveCheckpoint(
            nTP = self.nTP, nFP = self.nFP, nTN = self.nTN, nFN = self.nFN)
