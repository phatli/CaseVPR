#!/usr/bin/env python
"""
    This is a script to read the test results from the output folder and write them to "output/overall_results.csv"
"""
import sys
import argparse
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from casevpr.utils import CASEVPR_ROOT_DIR
if CASEVPR_ROOT_DIR not in sys.path:
    sys.path.insert(0, CASEVPR_ROOT_DIR)

from casevpr.utils import read_test_results
from scripts.configs.ds_configs import ds_info_dict

known_ds_names = list(ds_info_dict.keys())


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Read the test results')
    parser.add_argument('-c','--clean', action='store_true', help="Clean the folders which doesn\'t have results.json")

    # Parse the input arguments
    args = parser.parse_args()
    read_test_results(clean=args.clean, known_ds_names=known_ds_names)

if __name__ == '__main__':
    main()