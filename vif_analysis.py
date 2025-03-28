import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def calculate_vif(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    df_numeric = sm.add_constant(df_numeric)  # Add constant term

    vif = pd.DataFrame()
    vif["Features"] = df_numeric.columns
    vif["VIF Factor"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]

    return vif
