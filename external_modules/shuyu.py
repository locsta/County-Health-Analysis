"""external modules created by Shuyu"""
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import pandas as pd
import numpy as np
import datetime
from scipy import stats
import itertools # for combinations
from statsmodels.stats.power import TTestIndPower, TTestPower
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
def fix_missing(df):
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
