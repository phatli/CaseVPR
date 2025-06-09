import os
CASEVPR_ROOT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-4])

from .redis_utils import Mat_Redis_Utils
from .variable_dict import AttributeDict
from .time_probe import TimeProbe
from .vis_seq import vis_seq
from .default_args_vgt import ARGS
from .read_test_results import read_test_results, is_test_done
