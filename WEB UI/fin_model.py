# %% [markdown]
# # Introduction
# 
# In this notebook, I apply the complete data analysis process with machine learning to predict the EPS in binary format indicating if the input date is good or bad EPS which giving us an initial idea of the stock's quality prior going deep analysis for such stock.
# 
# ### __Data analysis process:__
# 
# __Data preprocessing__
# - Rename columns
# - Handle missing values
# - Check multicollinearity
# 
# __EDA__
# - Adjusted inflation rate on net income and market cap to illustrate the real value of the company
# - Inspect the cash flow
# - Inspect the average market cap in each industry
# - Inspect the average in every column in each industry
# - Determine the target variable for machine learning model
# - Handle outliers in the target variable
# 
# __Feature Engineering__
# - Create the target variable in the form of a binary classification
# - Feature Selection with machine learning model
# 
# __Modeling__
# - Apply classification models
# - Hyperparameter tuning with GridSearchCV
# 
# __Evaluation__
# - Feature importance
# - Confusion matrix
# - Accuracy
# - Classification report
# - ROC curve
# 
# __Summary__
# 
# Finally I have summarized the finding, assumption and further work to do and improve this notebook.

# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# %%
# Set the float format of pandas output for this notebook
# pd.options.display.float_format = '{:.3f}'.format
pd.options.display.max_columns = 100

# %%
# df = pd.read_csv('/kaggle/input/financial-statements-of-major-companies2009-2023/Financial Statements.csv')
df = pd.read_csv('C:\\Users\\cheta\\Documents\\FinGPT[1]\\Financial Statements.csv')
df.head()

# %% [markdown]
# # Data Preprocessing

# %%
df1 = df.copy()

# %%
# Rename columns

df1.rename(columns={'Company ':'Company'}, inplace=True)
df1.rename(columns={'Market Cap(in B USD)':'Market_cap'}, inplace=True)
df1.rename(columns={'Inflation Rate(in US)':'Inflation_rate'}, inplace=True)
df1.rename(columns={'Category':'Industry'}, inplace=True)
df1['Industry'] = df1['Industry'].replace('BANK', 'Bank')
df1.columns = df1.columns.str.replace(' ', '_')

# %%
# Handle missing values

print(f'Before: {df1.isna().sum()}')
df1['Market_cap'].fillna(value= df1['Market_cap'].mean(), inplace=True)
print("*"*100)
print(f'After: {df1.isna().sum()}')

# %%
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

# %% [markdown]
# ## Result of VIF
# 
# As a rule of thumb, without domain knowledge, VIF above 5 could indicate a high collinear with the other variables resulting in overfitting models. However, in this scenario, the financial ratios and some variables are derived from similar formulas and calculation. Therefore, we will remain all and handle the multicollinearity with dimension reduction methods.

# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %%
df1.info()

# %%
df1.describe()

# %%
# Draw a sample for the lowest ROE

df1.loc[ (df1['ROE']== np.min(df1['ROE']))]

# %%
# Adjust the net income and market cap after the inflation rate

def adjusted_inflation(df, net_income, market_cap, inflation):
    '''
    Get the adjusted net income and market cap
    '''
    df['Adjusted_net_income'] = df[net_income]/(1+df[inflation]/100)
    df['Adjusted_market_cap'] = df[market_cap]/(1+df[inflation]/100)
    return df

# %%
df1 = df1.sort_values('Year').reset_index(drop='index')
adjusted_inflation(df1, 'Net_Income', 'Market_cap', 'Inflation_rate')

df1.head()

# %%
# Check the value of these 2 variables to set the range in the plot.
df1[['Adjusted_net_income','Earning_Per_Share']].describe()

# %%
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

# %% [markdown]
# ## Do companies operate the cash flow effectively?
# 
# Cash flow margin can help to answer this question.
# 
# Based on the formula: $CF margin = \frac{cash flow}{net sales}$
# 
# Since there is no net sales available in this dataset, Cash flow from operating is used as it could represent sales and expenses.
# 

# %%
df1['CF_margin'] = df1['Cash_Flow_from_Operating'] / df1['Revenue']

# %%
# By Company
fig = px.histogram(df1, x='Company', y='CF_margin', color='Year',barmode='group', title='Cash Flow Margin by Company', labels={'CF_margin': 'Cash Flow Margin'})
fig.show()

# By Year
fig = px.bar(df1, x='Year', y='CF_margin', color='Company',barmode='group', title='Cash Flow Margin by Year', labels={'CF_margin': 'Cash Flow Margin'})
fig.show()

# By Industry
fig = px.histogram(df1, x='Industry', y='CF_margin', color='Company',barmode='group', title='Cash Flow Margin by Industry', labels={'CF_margin': 'Cash Flow Margin'})
fig.show()

# %%
with pd.option_context('display.max_rows', 1000):
    print(df1.groupby(["Industry", "Year"])["CF_margin"].mean())

# %%
import seaborn as sns

corr_df = pd.get_dummies(df1, drop_first=True)
corr_df['CF_margin'] = corr_df['Cash_Flow_from_Operating'] / corr_df['Revenue']
corr_df.corr()["CF_margin"].sort_values(ascending=False)

corr = np.around(corr_df.corr(), decimals=2)
plt.figure(figsize=(20, 16))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, annot_kws={"size": 8},linecolor="white")
plt.title("Correlation Heatmap")
plt.show()

# %%
# Average market cap in each industry

industries = df1['Industry'].unique()

for industry in industries:
    industry_df = df1[df1['Industry'] == industry]
    industry_df = industry_df.groupby('Year')['Market_cap'].mean()
    
    fig = px.line(industry_df, x=industry_df.index, y=industry_df.values, title=f'Average Market Cap for {industry}')
    fig.update_layout(yaxis_title='Market Cap')
    fig.show()

# %% [markdown]
# ## The average of each columns group by industry

# %%
df1_mean_by_category = df1.select_dtypes(include=['float64','int64'])
df1_mean_by_category = df1_mean_by_category.drop(columns=['mk_int'])
df1_mean_by_category = df1_mean_by_category.groupby(df1['Industry']).mean()
df1_mean_by_category

# %%
fig = px.scatter_3d(df1, x='Earning_Per_Share', y='ROE', z='ROA', color='Company', symbol='Year', opacity=0.7)
fig.show()

# %% [markdown]
# ## Selecting a target variable
# 
# The idea is that we are looking for good stocks to invest based on fundamental analysis. Therefore, we have several options with the following reasons:
# 
# - EPS can gauge profitability and growth also meaningful for investors.
# - Net Income can represent profitability.
# - Market Cap can demonstrate that the business is growing.
# - Free Cash Flow per Share can desmonstrate ability to generate cash and be torelent for unexpected circumstances.
# 
# In this notebook, __EPS__ is selected.

# %%
df1.head()

# %%
# Drop the adjusted columns because we will keep the inflation rate and original data. Let's see if models can perform well.
cleaned_df = df1.drop(columns=['mk_int','Adjusted_net_income','Adjusted_market_cap','CF_margin'])

# %%
cleaned_df.info()

# %%
# Handle outliers
num_types = cleaned_df.select_dtypes(['int64','float64'])

# Visualize with box plots and histogram
fig = px.box(num_types)
fig.show()

fig = px.box(cleaned_df, x='Earning_Per_Share')
fig.show()

fig = px.histogram(cleaned_df, x="Earning_Per_Share")
fig.show()

# %%
# %%timeit
# cleaned_df.drop(cleaned_df.query('Earning_Per_Share < -20').index)
# 2.78 ms ± 26.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
cleaned_df.drop(cleaned_df.loc[cleaned_df['Earning_Per_Share']< -20].index, inplace=True)
# 797 µs ± 12.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
fig = px.box(cleaned_df, x='Earning_Per_Share')
fig.show()

# %% [markdown]
# ## Feature Engineering
# 
# The target variable, Earning_Per_Share ('EPS'), is continuous. It needs to transform the continuous variable into in binary (0,1). For doing this, we need to set a threshold for the EPS to determine if each record will fall in 0 or 1.
# 
# ### Create threshold for target variable classification
# 
# This is a binary classification with machine learning approach, Here is my approach:
# 1. Standardize the data with scaling method
# 1. Apply Dimension Reduction method and force the method to reduce into 2 components (features)
# 1. Employ K-means clustering and visualize the result to conclude if it looks good
# 1. Check Check imbalance to deal with bias for modeling

# %%
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

cleaned_df = pd.get_dummies(cleaned_df,drop_first=True)

X = cleaned_df.drop(columns=['Earning_Per_Share'])

# Scaling
rs = RobustScaler()
X_scaled = rs.fit_transform(X)

# Dimension Reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_tsne)

plt.figure(figsize=(12, 8))

sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette='viridis', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('t-SNE and K-means Clustering with EPS')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()

# %%
cluster_df = pd.DataFrame({'EPS_Label': clusters}, index=cleaned_df.index)
print(f'Number of each label:\n{cluster_df.value_counts()}')
print("*"*100)

# Add the label to the main dataset
cleaned_df['EPS_label'] = cluster_df['EPS_Label']

# %%
# Check imbalance classes
imbalance_df = cleaned_df['EPS_label'].value_counts().reset_index()
fig = px.bar(imbalance_df, x='EPS_label', y='count',width=500, height=800)
fig.show()

# %% [markdown]
# # Feature Selection

# %%
# Feature Selection

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

method = []
acc = []

#define X and y
X = cleaned_df.drop(['Earning_Per_Share', 'Year', 'EPS_label'], axis=1)
y = cleaned_df['EPS_label']

#Method 1: using variance threshold
vt = VarianceThreshold(threshold=0.25)
vt.fit(X)
pd.DataFrame({'variance': vt.variances_,\
             'select_feature':vt.get_support()}, index = X.columns)
X_1 = X.iloc[:, vt.get_support()]
X_1.shape

#Method 2: Use SelectFromModel and LogisticRegression
method2 = SelectFromModel(estimator=LogisticRegression(max_iter=5000))
method2.fit(X,y)

pd.DataFrame({'coef':method2.estimator_.coef_[0],\
             'select_feature':method2.get_support()}, index=X.columns)
X_2 = method2.transform(X)
X_2.shape

#Method 3: Use Generalized Linear Model (Binomial) from statsmodels
X = X.astype(float) #If it is error, it may need to cast type float.
logit_model = sm.GLM(y,X,family=sm.families.Binomial())
result = logit_model.fit(fit_intercept=True)

#Select the features that coefficient value is greater than 0.02
X_3 = X.loc[:,abs(result.params)>=0.1]
X_3.shape

#Method 4: Use Recursive Feature Elimination (RFE) and Logistic Regression
rfe = RFE(estimator=LogisticRegression(max_iter=5000),\
         n_features_to_select=10, step=1)
rfe_result = rfe.fit(X,y)
X_4 = X.loc[:,rfe_result.support_]
X_4.shape

#Method 5: Use SelectFromModel and Random Forest classifier
method5 = SelectFromModel(estimator=RandomForestClassifier(random_state=42, n_estimators=100))
method5.fit(X, y)
X_5 = method5.transform(X)
X_5.shape

# %%
# Evaluate Feature Selection methods

# Performance of Feature Selection Method 1
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_1, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train1, y_train1)
y_pred_1 = logreg.predict(X_test1)
acc_1 = logreg.score(X_test1, y_test1)
method.append('Method 1')
acc.append(acc_1)

# Performance of Feature Selection Method 2
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_2, y, test_size=0.3, random_state=0)
logreg2 = LogisticRegression(max_iter=5000)
logreg2.fit(X_train2, y_train2)
y_pred_2 = logreg2.predict(X_test2)
acc_2 = logreg2.score(X_test2, y_test2)
method.append('Method 2')
acc.append(acc_2)

# Performance of Feature Selection Method 3
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_3, y, test_size=0.3, random_state=0)
logreg3 = LogisticRegression(max_iter=5000)
logreg3.fit(X_train3, y_train3)
y_pred_3 = logreg3.predict(X_test3)
acc_3 = logreg3.score(X_test3, y_test3)
method.append('Method 3')
acc.append(acc_3)

# Performance of Feature Selection Method 4
X_train4, X_test4, y_train4, y_test4 = train_test_split(X_4, y, test_size=0.3, random_state=0)
logreg4 = LogisticRegression(max_iter=5000)
logreg4.fit(X_train4, y_train4)
y_pred_4 = logreg4.predict(X_test4)
acc_4 = logreg4.score(X_test4, y_test4)
method.append('Method 4')
acc.append(acc_4)

# Performance of Feature Selection Method 5
X_train5, X_test5, y_train5, y_test5 = train_test_split(X_5, y, test_size=0.3, random_state=0)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train5, y_train5)
y_pred_5 = rf.predict(X_test5)
acc_5 = rf.score(X_test5, y_test5)
method.append('Method 5')
acc.append(acc_5)

# Print the feature selection methods in a table format
feature_selection_results = np.vstack( (method, acc)).T
feature_selection_results_df = pd.DataFrame(feature_selection_results, columns=['Method', 'Accuracy'])
feature_selection_results_df

# %%
# Method 5 (Model Random Forest) gives the best accuracy so we will use this feature selection.
# Check the total number of features from method 5
X_5.shape[1]

# %%
# Visualize the feature importance from Random Forest Feature Selection

feature_importances = method5.estimator_.feature_importances_
# Get the index
feature_idx = np.arange(len(feature_importances))
# Sort features based on importance
sorted_feat = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(feature_idx, feature_importances[sorted_feat], align="center")
plt.xticks(feature_idx, X.columns[sorted_feat], rotation='vertical')
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance from RandomForest Feature Selection")
plt.show()

# %% [markdown]
# ## Modeling

# %%
# Use the X and y at the time we employed the feature selection method
X_sel = X_5

X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)

# %%
# Apply Classification models
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# name of the algorithms to be using in modeling
names = ["Logististic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM",          
         "Decision Tree", "Naive Bayes", "Random Forest"]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel='linear'),
    SVC(kernel='rbf'),
    DecisionTreeClassifier(max_depth=3),
    GaussianNB(),
    RandomForestClassifier()
]

classifier_scores = {}

for classifier, name in zip(classifiers, names):
    pipeline = Pipeline(steps=[
                ('classifier', classifier)
        ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    score = np.mean(scores)
    classifier_scores[name] = score
    
scores_df = pd.DataFrame(list(classifier_scores.items()), columns=['Classifier', 'Accuracy'])
print(scores_df)

# Print the classifier with the best accuracy
best_classifier = max(classifier_scores, key=classifier_scores.get)
print()
print(f"Best Classifier: {best_classifier}")
print(f"Accuracy: {classifier_scores[best_classifier]}")

# %% [markdown]
# ## Hyperparameter tuning with GridSearchCV

# %%
import time

# start time
start_time_tuning = time.time()

best_score = 0
n_estimators_list = [100, 200, 300]
max_depth_list = [5, 10, 15]
min_samples_split_list = [2, 5, 10]
min_samples_leaf_list = [1, 2, 4]

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            for min_samples_leaf in min_samples_leaf_list:
                    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                                random_state=42)
                    scores = cross_val_score(rf, X_train, y_train, cv=5)
                    score = np.mean(scores)
                    if score > best_score:
                        best_score = score
                        best_hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 
                                            'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

print("Best score:", best_score)
print("Best hyperparameters:", best_hyperparams)

tuned_rf = RandomForestClassifier(**best_hyperparams, random_state=42)
tuned_rf.fit(X_train, y_train)

score_test = tuned_rf.score(X_test, y_test)
print("Test score:", score_test)

# end time
end_time_tuning = time.time()

# Calculate the runtime
runtime_tuning = end_time_tuning - start_time_tuning
print("Runtime for hyperparameter tuning:", runtime_tuning, "seconds")

# %% [markdown]
# # Evaluation

# %%
# Visualize the feature importance from Random Forest Classifier

feature_importances = tuned_rf.feature_importances_
# Get the index
feature_idx = np.arange(len(feature_importances))
# Sort features based on importance
sorted_feat = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(feature_idx, feature_importances[sorted_feat], align="center")
plt.xticks(feature_idx, X.columns[sorted_feat], rotation='vertical')
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance from Random Forest Classifier")
plt.show()

# %%
# Confusion Matrix

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score, confusion_matrix

y_pred = tuned_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, columns=np.unique(y_test),
                     index= np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (4,4))
sns.set(font_scale=1.2)
sns.heatmap(df_cm, annot=True, annot_kws={'size': 12},
           cbar=False, vmax=500, square=True, fmt="d", cmap="Reds")

# %%
# Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %%
# Classification Report

print(classification_report(y_test, y_pred))

# %%
# ROC

# Create a probability result for RandomForest
tuned_rf = RandomForestClassifier(**best_hyperparams)
tuned_rf.fit(X_train, y_train)
tuned_rf_prob = tuned_rf.predict_proba(X_test)[:,1]

# Create ROC
tuned_rf_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, tuned_rf_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='Tuned RandomForest' % tuned_rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(['ROC curve (AUC = {:.2f})'.format(roc_auc)], loc='lower right')
plt.show()
print("AUC:", roc_auc)

# %% [markdown]
# # Executive Summary
# - AAPL net income and EPS grew gradually with a consistent cash flow margin.
# - MFST had the best cash flow margin compared with the others in the IT industry.
# - AIG EPS was very fluctuated and cash flow margin was very low and negative in some years.
# - PYPL turned the EPS into positive in 2019 and plateau afterward.

# %% [markdown]
# # Modeling Result
# - Random forest is the best model with the accuracy at 0.98 after parameter tuning.
# - The evaluation showed the accuracy at 1.00 so the model might be overfitting although cross validation was employed. This issue may be caused by the small dataset and few companies for the modeling.
# - Net income is the most important feature with the score of 0.25 followed by EBITDA while share holder equity has the lowest score.


