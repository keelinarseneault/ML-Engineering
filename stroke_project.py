#%%[markdown]

# **Stroke Detection Project**

#%%

# Importing packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

# Load the data set
df = pd.read_csv('stroke dataset.csv')
target = 'stroke'
# %%

df.head()
df.info()
df.describe()

# Initial Observations

# -'id' column is randomly generated int digits, not useful for analysis   
# -Convert 'age' to int64 data type
# -'avg_glucose_level' has data type float64
# -'Stroke' column is the target variable (y)
# -Except for 'bmi' and 'avg_glucose_level', all other variables are categorical

#%%

# Drop ID column 
df.drop('id', axis = 1, inplace = True)
df.head()

#%% [markdown]

# Exploratory Data Analysis 

#%%

# Plots to show distributions and relationships between categorical and numeric variables

ax1 = sns.catplot(x="hypertension", y="bmi", hue="stroke", kind="violin", data=df, split=True)
plt.title('BMI Distribution by Hypertension Status')
plt.xlabel("Hypertension")
plt.ylabel("BMI")
ax1.set_xticklabels(['Yes', 'No'])
plt.show()

ax1 = sns.catplot(x="ever_married", y="bmi", hue="stroke", kind="violin", data=df, split=True)
plt.title('BMI Distribution by Marital Status')
plt.xlabel("Marital Status")
plt.ylabel("BMI")
ax1.set_xticklabels(['Married', 'Single'])
plt.show()

ax3 = sns.catplot(x="Residence_type", y="bmi", hue="stroke", kind="violin", data=df, split=True)
plt.title('BMI Distribution by Residence Type')
plt.xlabel("Residence Type")
plt.ylabel("BMI")
ax3.set_xticklabels(['Urban', 'Rural'])
plt.show()

ax4 = sns.catplot(y="bmi", x="smoking_status", hue="stroke", kind="violin", data=df, split=True)
plt.title('BMI Distribution by Smoking Status')
plt.xlabel("Smoking Status")
plt.ylabel("BMI")
ax4.set_xticklabels(['Former', 'Never', 'Current', 'Unknown'])
plt.show()

#%%

# Set random seed
random_seed = 42

# Set random seed in numpy
np.random.seed(random_seed)

#%%

from sklearn.model_selection import train_test_split

# Divide data into training (70%) and test (30%)
df_train, df_test = train_test_split(df, 
                                     train_size=0.7, 
                                     random_state=random_seed, 
                                     stratify=df[target])

# Divide test data into validation (50%) and test (50%)
df_val, df_test = train_test_split(df_test, 
                                   train_size=0.5, 
                                   random_state=random_seed, 
                                   stratify=df_test[target])

# Reset index
df_train, df_val, df_test = df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

print("\nDone.")

# %%

# Print dimensions of training, test, and validation data frames

# Training
print(pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns']))

# Test 
print(pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns']))

# Validation
print(pd.DataFrame([[df_val.shape[0], df_val.shape[1]]], columns=['# rows', '# columns']))


#%%

# Check for variable commonality between test, training, and validation data
from common_var_check import common_var_checker

df_common_var = common_var_checker(df_train, df_test, df_val, target)
print(df_common_var)

#%%

# Get features in training data but not in the validation or test data
uncommon_train_not_val_test = np.setdiff1d(df_train.columns, df_common_var['common var'])

# Print uncommon features
pd.DataFrame(uncommon_train_not_val_test, columns=['uncommon feature'])

#%%

# Get features in the test data but not in the training or validation data
uncommon_test_not_train_val = np.setdiff1d(df_test.columns, df_common_var['common var'])

# Print uncommon features
pd.DataFrame(uncommon_test_not_train_val, columns=['uncommon feature'])

#%%

# Get features in the validation data but not in the training or test data
uncommon_val_not_train_test = np.setdiff1d(df_val.columns, df_common_var['common var'])

# Print uncommon features
pd.DataFrame(uncommon_val_not_train_test, columns=['uncommon feature'])

#%% 

# Check for missing values

from nan_check import nan_check

df_nan = nan_check(df)
df_nan

df_miss =df_nan[df_nan['dtype'] == 'float64'].reset_index(drop=True)
df_miss

#%%

df['bmi'].isna().value_counts()     # 201 BMI observations are missing 

#%% 

# Impute missing data 

from sklearn.impute import SimpleImputer

# If there are missing values
if len(df_miss['var']) > 0:
    # The SimpleImputer
    si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # Impute the variables with missing values in df_train, df_val and df_test 
    df_train[df_miss['var']] = si.fit_transform(df_train[df_miss['var']])
    df_val[df_miss['var']] = si.transform(df_val[df_miss['var']])
    df_test[df_miss['var']] = si.transform(df_test[df_miss['var']])
   
print("\nDone.") 
#%%    

# Combine df_train, df_val and df_test
df = pd.concat([df_train, df_val, df_test], sort=False)

print("\nDone.")
#%%
# Call categorical variable checker function 

from cat_check import cat_var_checker
df_cat = cat_var_checker(df)

# Print the dataframe
df_cat    
    
#%%

# One-hot-encode the categorical features in the combined data
df = pd.get_dummies(df, columns=np.setdiff1d(df_cat['var'], [target]))

# Print the first 5 rows of df
df.head()

#%%

# Separating the training data
df_train = df.iloc[:df_train.shape[0], :]

# Separating the test data
df_test = df.iloc[df_train.shape[0] + df_val.shape[0]:, :]

# Separating the validation data
df_val = df.iloc[df_train.shape[0]:df_train.shape[0] + df_val.shape[0], :]

print("\nDone.")
#%%

# Print dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])

#%%

# Print the dimension of df_test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])

#%%

# Print the dimension of df_val
pd.DataFrame([[df_val.shape[0], df_val.shape[1]]], columns=['# rows', '# columns'])

#%%

# Get feature matrix
X_train = df_train[np.setdiff1d(df_train.columns, [target])].values
X_val = df_val[np.setdiff1d(df_val.columns, [target])].values
X_test = df_test[np.setdiff1d(df_test.columns, [target])].values

# Get target vector
y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

print("\nDone.")
#%%
# Standardize data
from sklearn.preprocessing import StandardScaler

# The StandardScaler
ss = StandardScaler()

# Standardize training data
X_train = ss.fit_transform(X_train)

# Standardize the test data
X_test = ss.transform(X_test)

# Standardize validation data
X_val = ss.transform(X_val)

print("\nDone.")
#%%

# Get class distribution 
pd.Series(y_train).value_counts()

# From initial analysis, the dataset is unbalanced. 
# There are 4699 cases of stroke = 0 and 209 cases of stroke = 1 among the participants. 
# This will require balancing the dataset (target variable) before any furthur tests or analysis.
# Balancing the data set using SMOTE: Synthetic Minority Oversampling Technique. Using this method, the minority target variable is oversampled using random values. 
# The technique uses the concept of K-NN or K neareast neighbors to intelligently generate synthetic data which resembles the values or shape of the outnumbered data instead of directly copying or reusing pre-existing values. 
# For more info: https://github.com/scikit-learn-contrib/imbalanced-learn

#%%

# Import SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=random_seed)

# Augment the training data
X_smote_train, y_smote_train = smote.fit_resample(X_train, y_train)

print("\nDone.")
# %%

# Get class distribution 

pd.Series(y_smote_train).value_counts()

#%% 

# Create dictionaries of models and pipelines

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

models = {'lr': LogisticRegression(class_weight='balanced', random_state=random_seed)}

pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])

print("\nDone.")
#%%

# Get the feature matrix and target vector in the combined training and validation data
# target vector in the combined training and validation data
# PredefinedSplit
from predefined_split import get_train_val_ps

X_train_val, y_train_val, ps = get_train_val_ps(X_smote_train, y_smote_train, X_val, y_val)

print("\nDone.")
#%%

# Create dictionary of parameter grids
param_grids = {}

# The parameter grid of tol
tol_grid = [10 ** -5, 10 ** -4, 10 ** -3]

# The parameter grid of C
C_grid = [10, 1, 0.1]

# Update param_grids
param_grids['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid}]

# Tune hyperparameters with GridSearchCV

from sklearn.model_selection import GridSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_params_estimator_gs = []

# For each model
for acronym in pipes.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_macro',
                      n_jobs=2,
                      cv=ps,
                      return_train_score=True)
        
    # Fit the pipeline
    gs = gs.fit(X_train_val, y_train_val)
    
    # Update best_score_params_estimator_gs
    best_score_params_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
    
    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    
    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score', 
                         'std_test_score', 
                         'mean_train_score', 
                         'std_train_score',
                         'mean_fit_time', 
                         'std_fit_time',                        
                         'mean_score_time', 
                         'std_score_time']
    
    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])

# Get the best_score, best_params and best_estimator obtained by GridSearchCV
best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]

# Test the best model selected on the test data

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

# Get the prediction on the testing data using best_model
y_test_pred = best_estimator_gs.predict(X_test)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)

# Get the auc
auc = roc_auc_score(y_test, y_test_pred)

# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore, auc]], columns=['Precision', 'Recall', 'F1-score', 'AUC'])

# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
   y_test, y_test_pred)

plt.show()
# %%
