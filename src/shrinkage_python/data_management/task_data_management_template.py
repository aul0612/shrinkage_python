"""Tasks for managing the data."""

import pandas as pd

from shrinkage_python.config import BLD, SRC
from shrinkage_python.data_management.stats4schools_smoking_template import (
    clean_stats4schools_smoking,
)


def task_clean_stats4schools_smoking_data(
    script=SRC / "data_management" / "stats4schools_smoking_template.py",
    data=SRC / "data" / "stats4schools_smoking_template.csv",
    produces=BLD / "data" / "stats4schools_smoking.pickle",
):
    """Clean the stats4schools smoking data set."""
    data = pd.read_csv(data)
    data = clean_stats4schools_smoking(data)
    data.to_pickle(produces)
