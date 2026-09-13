import sys
import datetime
import re
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
import pandas as pd
import numpy as np

from main import FinancialDataReconciler, DATASET_DIR

reconciler = FinancialDataReconciler(DATASET_DIR)
samples = pd.read_csv(DATASET_DIR / "sample_requests.csv")

print("Loaded reconciler and samples.")
