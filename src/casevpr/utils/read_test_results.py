import os
import json
import pandas as pd
import shutil
import re

from . import CASEVPR_ROOT_DIR

known_ds_names_default = ['oxford', 'oxford_v', 'nordland', 'rosimg']


def read_test_results(**kwargs):
    """
        Read the test results from the output folder and write them to "output/overall_results.csv"
    """
    clean_empty_folder = kwargs.get('clean', False)
    known_ds_names = kwargs.get('known_ds_names', known_ds_names_default)

    # Set the directory path where test results are stored
    result_dir = os.path.join(CASEVPR_ROOT_DIR, 'output/test_logs')

    # Create an empty list to store the results
    results = []

    # Loop through all the subdirectories in the result directory
    for subdir in os.listdir(result_dir):
        subdir_path = os.path.join(result_dir, subdir)
        if os.path.isdir(subdir_path):
            result_file = os.path.join(subdir_path, 'result.json')
            if os.path.isfile(result_file):
                with open(result_file) as f:
                    # Load the json file
                    result_data = json.load(f)

                    # Extract the properties from the directory name
                    test_info = os.path.basename(subdir_path)
                    test_date, dataset1_name, dataset2_name, frontend_name, backend_name, pre_name, loop_number, additional_info, dataset_name = get_test_info(
                        test_info, known_ds_names=known_ds_names)

                    # Create a dictionary of the properties and the result data
                    if '+' in backend_name:
                        if len(result_data) != 2:
                            if "precision" in result_data.keys():
                                result_data["precision"] *= 100
                            if "accuracy" in result_data.keys():
                                result_data["accuracy"] *= 100
                            if "recall" in result_data.keys():
                                result_data["recall"] *= 100
                            bn = backend_name.split('+')[0]
                            test_result = {'test_date': test_date,
                                           'dataset_name': dataset_name,
                                           'Loop1_name': dataset1_name,
                                           'Loop2_name': dataset2_name,
                                           'frontend_name': frontend_name,
                                           'backend_name': bn,
                                           'pre_name': pre_name,
                                           'loop_number': loop_number,
                                           'additional_info': additional_info}
                            result_data = inject_sum(result_data)
                            result_data = inject_posdist(
                                result_data, additional_info, subdir_path)

                            result_data = inject_gt(
                                result_data, additional_info, subdir_path)
                            result_data = inject_distcosthres(
                                result_data, additional_info, subdir_path)
                            test_result.update(result_data)
                            test_result['folder_name'] = test_info
                            results.append(test_result)
                        else:
                            for bn in backend_name.split('+'):
                                bn_result = result_data[bn]
                                if "precision" in bn_result.keys():
                                    bn_result["precision"] *= 100
                                if "accuracy" in bn_result.keys():
                                    bn_result["accuracy"] *= 100
                                if "recall" in bn_result.keys():
                                    bn_result["recall"] *= 100
                                test_result = {'test_date': test_date,
                                               'dataset_name': dataset_name,
                                               'Loop1_name': dataset1_name,
                                               'Loop2_name': dataset2_name,
                                               'frontend_name': frontend_name,
                                               'backend_name': bn,
                                               'pre_name': pre_name,
                                               'loop_number': loop_number,
                                               'additional_info': additional_info}
                                bn_result = inject_sum(bn_result)
                                bn_result = inject_posdist(
                                    bn_result, additional_info, subdir_path)
                                bn_result = inject_gt(
                                    bn_result, additional_info, subdir_path)
                                bn_result = inject_distcosthres(
                                    bn_result, additional_info, subdir_path)
                                test_result.update(bn_result)
                                test_result['folder_name'] = test_info
                                results.append(test_result)
                    else:
                        if "precision" in result_data.keys():
                            result_data["precision"] *= 100
                        test_result = {'test_date': test_date,
                                       'dataset_name': dataset_name,
                                       'Loop1_name': dataset1_name,
                                       'Loop2_name': dataset2_name,
                                       'frontend_name': frontend_name,
                                       'backend_name': backend_name,
                                       'pre_name': pre_name,
                                       'loop_number': loop_number,
                                       'additional_info': additional_info}
                        result_data = inject_sum(result_data)
                        result_data = inject_posdist(
                            result_data, additional_info, subdir_path)
                        result_data = inject_gt(
                            result_data, additional_info, subdir_path)
                        result_data = inject_distcosthres(
                            result_data, additional_info, subdir_path)

                        test_result.update(result_data)
                        test_result['folder_name'] = test_info
                        results.append(test_result)
            else:
                if clean_empty_folder:
                    print(
                        f"Removing empty folder {subdir_path} - no result.json file found")
                    shutil.rmtree(subdir_path)
                else:
                    print(
                        f"Skipping {subdir_path} - no result.json file found")

    # Create a pandas dataframe from the results list
    df = pd.DataFrame(results)

    df = df.sort_values(
        by=["dataset_name", "Loop1_name", "Loop2_name", "backend_name"], ascending=True)
    grouped = df.groupby(["dataset_name", "Loop1_name", "Loop2_name"])

    # define a ranking function
    def rank_by_precision(group):
        group = group.sort_values(by=["precision"], ascending=False)
        group["rank"] = range(1, len(group) + 1)
        return group

    # apply the ranking function to each group
    ranked_df = grouped.apply(rank_by_precision)

    output_file_path = f"{CASEVPR_ROOT_DIR}/output/overall_results.csv"
    ranked_df.to_csv(output_file_path, index=False)
    print(f"Tests results saved to {output_file_path}")


def get_test_info(test_info, known_ds_names=known_ds_names_default):
    test_date, dataset1_name, dataset2_name, frontend_name, backend_name, pre_name, loop_number, additional_info = test_info.split(
        '-')

    found_knonw_ds = False
    for ds_name in known_ds_names:
        if dataset1_name.startswith(ds_name) and dataset2_name.startswith(ds_name):
            found_knonw_ds = True
            dataset_name = ds_name
            dataset1_name = dataset1_name.replace(
                f"{ds_name}_", '')
            dataset2_name = dataset2_name.replace(
                f"{ds_name}_", '')
            break
    assert found_knonw_ds, f"Unknown dataset_name {dataset1_name} and {dataset2_name} found, please to add this to known_ds_names"

    return test_date, dataset1_name, dataset2_name, frontend_name, backend_name, pre_name, loop_number, additional_info, dataset_name


def inject_sum(dic):
    dic["numQ"] = dic["nTP"] + dic["nFP"] + dic["nFN"] + dic["nTN"]
    return dic


def inject_posdist(dic, info, subdir_path):
    config_file = os.path.join(subdir_path, 'config.json')
    if os.path.exists(config_file):
        dic["pos_dist"] = json.load(open(config_file))[
            "seqbackend_params"]["positive_dist"]
    elif "pos_dist" in info:
        match = re.search(r"pos_dist(\d+)", info)
        dic["pos_dist"] = int(match.group(1))
    else:
        dic["pos_dist"] = 11
    return dic


def inject_gt(dic, info, subdir_path):
    config_file = os.path.join(subdir_path, 'config.json')
    if os.path.exists(config_file):
        if json.load(open(config_file))[
                "seqbackend_params"]["seq_gt"]:
            if json.load(open(config_file))["seqbackend_params"].get("seq_gt_vgt", False):
                dic["gt"] = "s2s_lax"
            else:
                dic["gt"] = "s2s_strict"
        else:
            dic["gt"] = "s2i"
    elif "seqgt" in info:
        if "vgtseqgt" in info:
            dic["gt"] = "s2s_lax"
        else:
            dic["gt"] = "s2s_strict"
    else:
        dic["gt"] = "s2i"
    return dic


def inject_distcosthres(dic, info, subdir_path):
    config_file = os.path.join(subdir_path, 'config.json')
    if os.path.exists(config_file):
        dic["dist_cos_thres"] = json.load(open(config_file))[
            "seqbackend_params"].get("dist_cos_thres", 1)
    else:
        dic["dist_cos_thres"] = 1
    return dic


def is_test_done(test_folder_name, seqbackend_params, seq_gt=False, seq_gt_vgt=False, numQ=None, known_ds_names=known_ds_names_default):
    try:
        print(f"Reading test results...")
        # Make sure this function is defined and necessary
        read_test_results(known_ds_names=known_ds_names)
        output_file_path = f"{CASEVPR_ROOT_DIR}/output/overall_results.csv"
        df = pd.read_csv(output_file_path)
        df.fillna('', inplace=True)

        print(f"Extracting test info...")
        test_date, dataset1_name, dataset2_name, frontend_name, backend_name, pre_name, loop_number, additional_info, dataset_name = get_test_info(
            test_folder_name, known_ds_names=known_ds_names)
        if "+" in backend_name:
            result = True
            for bn in backend_name.split("+"):
                info_dict = {
                    "test_date": test_date,
                    "dataset_name": dataset_name,
                    "Loop1_name": dataset1_name,
                    "Loop2_name": dataset2_name,
                    "frontend_name": frontend_name,
                    "backend_name": bn,
                    "pre_name": pre_name,
                    "loop_number": int(loop_number),
                    "additional_info": additional_info
                }
                info_dict["pos_dist"] = seqbackend_params["positive_dist"]
                info_dict["gt"] = "s2s_lax" if seq_gt and seq_gt_vgt else "s2s_strict" if seq_gt else "s2i"
                info_dict["dist_cos_thres"] = seqbackend_params['dist_cos_thres']
                result = result and is_test_in_df(numQ, info_dict, df)
        else:
            test_info = {
                'test_date': test_date,
                'dataset_name': dataset_name,
                'Loop1_name': dataset1_name,
                'Loop2_name': dataset2_name,
                'frontend_name': frontend_name,
                'backend_name': backend_name,
                'pre_name': pre_name,
                'loop_number': int(loop_number),
                'additional_info': additional_info,
            }
            test_info["pos_dist"] = seqbackend_params["positive_dist"]
            test_info["gt"] = "s2s_lax" if seq_gt and seq_gt_vgt else "s2s_strict" if seq_gt else "s2i"
            test_info['dist_cos_thres'] = seqbackend_params['dist_cos_thres']
            result = is_test_in_df(numQ, test_info, df)

        return result

    except Exception as e:
        print(f"Error while reading test results: {e}")
        return False

# TODO: Hard-coded values check should be replaced


def is_test_in_df(numQ, test_info, df):
    conditions = (
        (df["dataset_name"] == test_info["dataset_name"]) &
        (df["Loop1_name"] == test_info["Loop1_name"]) &
        (df["Loop2_name"] == test_info["Loop2_name"]) &
        (df["backend_name"] == test_info["backend_name"]) &
        (df["frontend_name"] == test_info["frontend_name"]) &
        (df["pre_name"] == test_info["pre_name"]) &
        (df["loop_number"] == test_info["loop_number"]) &
        (df["additional_info"] == test_info["additional_info"]) &
        (df["pos_dist"] == test_info["pos_dist"]) &
        (df["gt"] == test_info["gt"]) &
        (df["dist_cos_thres"] == test_info["dist_cos_thres"])
    )
    result = df[conditions]
    if not result.empty and numQ is not None:
        result = result[result["numQ"] == numQ]

    print(f"Found {len(result)} results existed.")
    return not result.empty
