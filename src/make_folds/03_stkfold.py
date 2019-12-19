import os
import sys
import pandas as pd

from validate_utils import FoldValidation

import warnings
warnings.filterwarnings('ignore')

# ===============
# Settings
# ===============
fname = os.path.basename(sys.argv[0])
TRAIN_PATH = f'../data/input/train_data.feather'
OUTPUT_PATH = f'../folds/{fname.split(".")[0]}.feather'
N_FOLD = 5


# ===============
# Main
# ===============
df = pd.read_feather(TRAIN_PATH)
fold_validation = FoldValidation(df, stratify_arr=df['area'], fold_num=N_FOLD)
folds = fold_validation.make_split(valid_type='StratifiedKFold')
folds.to_feather(OUTPUT_PATH)
