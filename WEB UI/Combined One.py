import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# Set the float format of pandas output for this notebook
# pd.options.display.float_format = '{:.3f}'.format
pd.options.display.max_columns = 100


# df = pd.read_csv('/kaggle/input/financial-statements-of-major-companies2009-2023/Financial Statements.csv')
df = pd.read_csv('C:\\Users\\cheta\\Documents\\FinGPT[1]\\Financial Statements.csv')
df.head()


df1 = df.copy()


# Rename columns

df1.rename(columns={'Company ':'Company'}, inplace=True)
df1.rename(columns={'Market Cap(in B USD)':'Market_cap'}, inplace=True)
df1.rename(columns={'Inflation Rate(in US)':'Inflation_rate'}, inplace=True)
df1.rename(columns={'Category':'Industry'}, inplace=True)
df1['Industry'] = df1['Industry'].replace('BANK', 'Bank')
df1.columns = df1.columns.str.replace(' ', '_')


# Handle missing values

print(f'Before: {df1.isna().sum()}')
df1['Market_cap'].fillna(value= df1['Market_cap'].mean(), inplace=True)
print("*"*100)
print(f'After: {df1.isna().sum()}')


# Check Multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Drop non-numeric columns
numeric_cols = df1.select_dtypes(include=[np.number]).columns
df_numeric = df1[numeric_cols]

# Add a constant term as needed for VIF calculation
df_numeric = sm.add_constant(df_numeric)

vif = pd.DataFrame()
vif["Features"] = df_numeric.columns
vif["VIF Factor"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]

print(vif)


# Draw a sample for the lowest ROE

df1.loc[ (df1['ROE']== np.min(df1['ROE']))]


# Adjust the net income and market cap after the inflation rate

def adjusted_inflation(df, net_income, market_cap, inflation):
    '''
    Get the adjusted net income and market cap
    '''
    df['Adjusted_net_income'] = df[net_income]/(1+df[inflation]/100)
    df['Adjusted_market_cap'] = df[market_cap]/(1+df[inflation]/100)
    return df


df1 = df1.sort_values('Year').reset_index(drop='index')
adjusted_inflation(df1, 'Net_Income', 'Market_cap', 'Inflation_rate')

df1.head()


# Check the value of these 2 variables to set the range in the plot.
df1[['Adjusted_net_income','Earning_Per_Share']].describe()


import matplotlib.pyplot as plt
import plotly.express as px

%matplotlib inline

df1['mk_int'] = df1['Adjusted_market_cap'].round().astype(int)

# Scatter plot with animation
fig = px.scatter(df1, x='Adjusted_net_income',y='Earning_Per_Share', color='Company', size="mk_int", size_max=40,
              animation_frame="Year", animation_group="Company", range_x=[-13000,100000], range_y=[-1,15])
fig.show()

# Line chart
fig = px.line(df1, x='Year', y='Adjusted_net_income', color='Company',title='Adjusted Net Income')
fig.show()

# Bar chart
fig = px.bar(df1, x='Year', y='Earning_Per_Share', color='Company', barmode="group", title='EPS')
fig.show()