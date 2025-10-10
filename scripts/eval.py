#!/usr/bin/env python

"""
This script is used to automatically do offline tests on open-source dataset without using ROS environment.
To run this code, install this package in root using "pip install -e .".
"""
import sys
import os
import argparse
import json
import multiprocessing as mp
import copy
import queue
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import subprocess  # for querying GPU usage
import time       # for GPU memory check


CASEVPR_ROOT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
if CASEVPR_ROOT_DIR not in sys.path:
    sys.path.insert(0, CASEVPR_ROOT_DIR)



GPU_IDS = []
WORKER_GPU = None
MSG_QUEUE = None

def _init_worker(gpu_ids, msg_queue):
    """
    Runs once in each child when the pool starts.
    Records the GPU list, shared queue, and routes stderr through tqdm.
    """
    global GPU_IDS, WORKER_GPU, MSG_QUEUE
    GPU_IDS = gpu_ids
    MSG_QUEUE = msg_queue
    identity = mp.current_process()._identity
    idx = identity[0] - 1 if identity else 0

    assigned = gpu_ids[idx % len(gpu_ids)]
    os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned)
    WORKER_GPU = assigned
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    import torch
    torch.cuda.set_device(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='scripts/configs/batch_tests.json',
                        help='Path to the test JSON file.')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--no_cache', action='store_true', help='run without feature cache')
    parser.add_argument('--ignore_ckpt', action='store_true', help='run without 1st loop ckpt / Run 1st loop again')
    parser.add_argument('--show_console', action='store_true', help='Show console output')
    parser.add_argument('--process_num', type=int, default=1, help='Number of processes to run in parallel.')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0],
                        help='The GPU IDs to use (one or more).')
    return parser.parse_args()

def update_dict(orig_dict, update_dict):
    new_dict = copy.deepcopy(orig_dict)
    def apply_updates(target_dict, updates):
        for key, value in updates.items():
            if key in target_dict and isinstance(value, dict) and isinstance(target_dict[key], dict):
                apply_updates(target_dict[key], value)
            else:
                target_dict[key] = value
    apply_updates(new_dict, update_dict)
    return new_dict

def main():
    args = get_args()
    global GPU_IDS, MSG_QUEUE
    GPU_IDS = args.gpu_ids

    debug = args.debug
    use_cache = not args.no_cache
    ignore_ckpt = args.ignore_ckpt
    processes_num = args.process_num

    JSON_PATH = args.json_path if args.json_path.startswith('/') else os.path.join(CASEVPR_ROOT_DIR, args.json_path)

    # Load test setting
    with open(JSON_PATH, 'r') as file:
        tests_configs = json.load(file)

    test_lst = tests_configs['test_lst']
    pipeline_lst = tests_configs['pipeline_lst']
    default_settings = tests_configs['default_settings']

    # build task list
    tasks = []
    showConsole = True if debug or (processes_num == 1 and args.show_console) else False
    for settings_changes in tests_configs['settings_lst']:
        settings = update_dict(default_settings, settings_changes)
        seq_gt = settings['seq_gt']
        seq_gt_vgt = settings['seq_gt_vgt']
        save_seq = settings['save_seq']
        save_retrieval = settings.get('save_retrieval', False)
        save_feature_cache = settings.get('save_feature_cache', True)
        test_name = settings['test_name']
        seperate_ds = settings['seperate_ds']
        default_seqbackend_params = settings['default_seqbackend_params']
        search_nonkeyframe = default_seqbackend_params.get("search_nonkeyframe", False)

        for pl in pipeline_lst:
            for fe in pl["fe_lst"]:
                for be in pl["be_lst"]:
                    for ds_test in test_lst:
                        ds_name = ds_test["ds_name"]
                        for l1 in ds_test["l1"]:
                            tasks.append((
                                debug, use_cache, ignore_ckpt,
                                seq_gt, seq_gt_vgt, save_seq,
                                save_retrieval, save_feature_cache,
                                test_name, seperate_ds,
                                default_seqbackend_params,
                                search_nonkeyframe, ds_test,
                                ds_name, fe, be, l1,
                                showConsole
                            ))

    # setup manager & message queue
    manager = mp.Manager()
    msg_queue = manager.Queue()
    MSG_QUEUE = msg_queue

    if debug or processes_num == 1:
        os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[0])
        import torch
        torch.cuda.set_device(0)
        for task in tasks:
            proceed_task(task)
    else:
        with mp.Pool(
            processes=processes_num,
            initializer=_init_worker,
            initargs=(args.gpu_ids, msg_queue)
        ) as pool:
            results = pool.imap(proceed_task, tasks)
            pbar = tqdm(
                total=len(tasks),
                desc="Running tests",
                position=0,
                leave=True
            )
            for _ in range(len(tasks)):
                # flush worker messages first
                while True:
                    try:
                        msg = msg_queue.get_nowait()
                        pbar.write(msg)
                    except queue.Empty:
                        break
                    
                next(results)
                pbar.update()

            # flush any remaining messages
            while True:
                try:
                    msg = msg_queue.get_nowait()
                    pbar.write(msg)
                except queue.Empty:
                    break


def proceed_task(args):

    (debug, use_cache, ignore_ckpt, seq_gt, seq_gt_vgt,
     save_seq, save_retrieval, save_feature_cache,
     test_name, seperate_ds, default_seqbackend_params,
     search_nonkeyframe, ds_test, ds_name,
     fe, be, l1, showConsole) = args

    from casevpr import LoopDetector, get_testName
    from casevpr.utils import read_test_results, is_test_done
    from scripts.configs.ds_configs import ds_info_dict

    known_ds_names = list(ds_info_dict.keys())

    # import torch
    gpu_to_use = WORKER_GPU if WORKER_GPU is not None else GPU_IDS[0]
    # torch.cuda.set_device(gpu_to_use)

    # send startup message
    def log(msg):
        if MSG_QUEUE is not None:
            MSG_QUEUE.put(msg)
        print(msg)

    start_msg = f">> [RUNNING] Dataset: {ds_name}_{l1} Pipeline: {fe}_{be} GPU_ID: {gpu_to_use}"
    log(start_msg)

    # wait until at least 3GB free on this GPU
    waiting_msg_shown = False
    while True:
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,nounits,noheader",
                "-i", str(gpu_to_use)
            ]).decode().strip()
            free_mem = int(out.splitlines()[0])
        except Exception:
            free_mem = 0
        if free_mem >= 3000:
            break
        if not waiting_msg_shown:
            wait_msg = f">> Waiting for GPU {gpu_to_use} to have >=3GB free (current {free_mem}MB)"
            log(wait_msg)
            waiting_msg_shown = True
        time.sleep(5)

    try:
        dist_cos_thres = default_seqbackend_params.get("dist_cos_thres", 0.7)
        ds_info = ds_info_dict[ds_name]
        if ignore_ckpt or not hasCP(fe, f"{ds_name}_{l1}", dist_cos_thres):
            default_seqbackend_params["backend_name"] = "seqslam" if "seq_desc" not in be else "seq_desc"
            default_seqbackend_params["seq_gt"] = False
            default_seqbackend_params["search_nonkeyframe"] = False
            ds = ds_info["class"](l1, ds_info["path"])
            ld = LoopDetector(
                f"{ds_name}_{l1}", f"{ds_name}_{ds_test['l2'][0]}", True,
                model_name=fe, seqbackend_params=default_seqbackend_params,
                save_feature_cache=save_feature_cache, debug=debug,
                save_result_every_frame=False, showConsole=showConsole,
                only_log_vectors=True
            )
            for i in range(ds.len):
                img, pose = ds[i]
                ld.callback(img, pose)
            if not debug:
                ld.saveResult()
            ld.saveSeqCheckpoint()
        else:
            print(f">> SKIP 1st loop for {ds_name}_{l1}")

        default_seqbackend_params["backend_name"] = be
        for l2 in ds_test["l2"]:
            default_seqbackend_params["seq_gt"] = seq_gt
            default_seqbackend_params["search_nonkeyframe"] = search_nonkeyframe
            default_seqbackend_params["positive_dist"] = ds_test.get(
                "positive_dist", default_seqbackend_params["positive_dist"]
            )
            if seq_gt:
                default_seqbackend_params["seq_gt_vgt"] = seq_gt_vgt
            if l1 == l2 and ds_test.get("skip_same", False):
                continue
            ds2 = ds_info["class"](l2, ds_info["path"])
            ds_name_1 = f"{ds_name}_{l1}"
            ds_name_2 = f"{ds_name}_{l2}"
            testName = get_testName(
                ds_name_1, ds_name_2, False, fe, test_name, default_seqbackend_params
            )
            if is_test_done(testName, default_seqbackend_params, seq_gt, seq_gt_vgt, known_ds_names=known_ds_names):
                print(f">> SKIP existing {testName}")
            else:
                print(f">> RUNNING {testName}")
                ld = LoopDetector(
                    ds_name_1, ds_name_2, False, model_name=fe,
                    seqbackend_params=default_seqbackend_params,
                    save_seq=save_seq, test_name=test_name,
                    seperate_ds=seperate_ds, debug=debug,
                    save_retrieval=save_retrieval,
                    save_feature_cache=save_feature_cache,
                    save_result_every_frame=False,
                    showConsole=showConsole
                )
                for i in range(ds2.len):
                    img, pose = ds2[i]
                    ld.callback(img, pose, use_cache=use_cache)
                if not debug:
                    ld.saveResult()
                read_test_results(known_ds_names=known_ds_names)
    except Exception:
        error_msg = f"❌ ERROR Dataset: {ds_name}_{l1} Pipeline: {fe}_{be} GPU_ID={gpu_to_use}"
        log(error_msg)
        raise


def hasCP(model_name, dataset_name, dist_cos_thres):
    save_path = os.path.join(CASEVPR_ROOT_DIR, 'output', 'ckpts')
    saved_file = os.path.join(
        save_path,
        f"{model_name}-distcos{str(dist_cos_thres).replace('.','')}-{dataset_name}.pkl"
    )
    return os.path.exists(saved_file)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
