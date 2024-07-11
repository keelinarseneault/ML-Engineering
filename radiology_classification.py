#%% [markdown]

### **Logistic Regression Model to Detect Cancer Malignancy**

# Features data from digitized images and the characteristics of cell nuclei present in an observed mass. 
#
#Ten real-valued features are computed for each cell nucleus:
#
# a) radius (mean of distances from center to points on the perimeter)
#
# b) texture (standard deviation of gray-scale values)
#
# c) perimeter
#
# d) area
#
# e) smoothness (local variation in radius lengths)
#
# f) compactness (perimeter^2 / area - 1.0)
#
# g) concavity (severity of concave portions of the contour)
#
# h) concave points (number of concave portions of the contour)
#
# i) symmetry
#
# j) fractal dimension ("coastline approximation" - 1)
#
# The mean, standard error and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features. For example, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

# %%

# Importing packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("\nDone.")

#%%

# Load the data set

df = pd.read_csv('radiology.csv')
target = 'diagnosis'

print("\nDone.")
# %%

# View dataframe and data types
df.head()
df.info()

#%%

# Drop ID and 'Unnamed' column 
df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
df.head()

# %% [markdown]

### Data Preprocessing

#%%

# Check for missing values
from nan_check import nan_check

df_nan = nan_check(df)
df_nan

df_miss =df_nan[df_nan['dtype'] == 'float64'].reset_index(drop=True)
df_miss

#%%

# Visualize distributions of the mean features

for i in ('radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'):
    df[i].plot.box(title = i)
    plt.show()
    
#%%

df = df[(np.abs(stats.zscore(df.select_dtypes(include=np.number))) < 3).all(axis=1)]
df

#%%

#%%

# Visualize class imbalance
print(df['diagnosis'].value_counts())
df['diagnosis'].value_counts().plot(kind='bar')
plt.title("Class Imbalance in Dataset")
plt.show()

#%%

### Visualize possible multicollinearity 
f = plt.figure(figsize=(20, 16))
plt.matshow(df.corr(), fignum=f.number)
plt.title('Correlation Matrix', fontsize = 50, x = 0.5, y = -0.1)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=20, rotation=90)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=20)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=22)
plt.show


#%%

# Call categorical variable checker function to see if categorical features are present for one-hot-encoding
from cat_check import cat_var_checker
df_cat = cat_var_checker(df)

# Print the dataframe
df_cat    

#%%

# Encode target variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encode categorical target in the combined data
df[target] = le.fit_transform(df[target])

# Print the first 5 rows 
df.head()

#%% 

# Set random seed
random_seed = 42

#%%

# Split into training and test data
from sklearn.model_selection import train_test_split

# Divide data into training (70%) and test (30%)
df_train, df_test = train_test_split(df,
                                     train_size=0.7, 
                                     random_state=random_seed,
                                     stratify=df[target])

#%%

# Show size of training data
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])

#%%

# Show size of test data
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])

#%%

# Split feature and target
X_train = df_train[np.setdiff1d(df_train.columns, [target])].values
X_test = df_test[np.setdiff1d(df_test.columns, [target])].values
y_train = df_train[target].values
y_test = df_test[target].values

#%%

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay

#%% 

# Scale data
sc = StandardScaler()
scaled_x_train = pd.DataFrame(
    sc.fit_transform(X_train))

#%%

# Determine number of components for PCA
from sklearn.decomposition import PCA
pca = PCA().fit(scaled_x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Proportion of Variance Explained by Components")
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')


#%%

# Construct Pipeline

pipe = Pipeline(steps = [('standardscaler', StandardScaler()),
                      ('pca', PCA(n_components=6)),
                      ('logisticregression', LogisticRegression())])

model = pipe.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

#%%
pd.DataFrame([[precision, recall, fscore, auc]], columns=[ 'Precision', 'Recall','F1-score', 'AUC'])

#%%
ConfusionMatrixDisplay.from_predictions(
   y_test, y_pred)
plt.title('Confusion Matrix')
