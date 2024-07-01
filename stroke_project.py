#%%[markdown]

# Stroke Detection Project

#%%
# Importing packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks")

#%%
# Loading the data set
df = pd.read_csv('stroke dataset.csv')
target = 'stroke'
# %%
df.info()
df.head()

# Starting initial EDA 
# -'id' column is randomly generated int digits, not required.    
# -Converting 'age' column to type int64. 
# -'avg_glucose_level' is of type float64.
# -'Stroke' column is the target variable or y. 
# -Except for 'bmi' and 'avg_glucose_level', all other columns are categorical in nature.

#%%
# EDA - Box plots to show relationships between stroke and work-type & stroke and marriage status 

ax1 = sns.boxplot(x="ever_married", y="bmi", color="b", data=df)
plt.title('BMI Distribution by Marital Status')
ax1.set_xticklabels(['Yes', 'No'])
plt.show()

print("\nReady to continue.")

ax3 = sns.boxplot(x="Residence_type", y="bmi", color="b", data=df)
plt.title('BMI Distribution by Residence Type')
ax3.set_xticklabels(['Urban', 'Rural'])
plt.show()

print("\nReady to continue.")
#%%
	
from sklearn.model_selection import train_test_split

# Divide the data into training (60%) and test (40%)
df_train, df_test = train_test_split(df, 
                                     train_size=0.6, 
                                     random_state=random_seed, 
                                     stratify=df[target])

# Divide the test data into validation (50%) and test (50%)
df_val, df_test = train_test_split(df_test, 
                                   train_size=0.5, 
                                   random_state=random_seed, 
                                   stratify=df_test[target])

# Reset the index
df_train, df_val, df_test = df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

# %%

# Drop 'id' column 

df.drop('id', axis = 1, inplace = True)

#%% 

# Check for missing values

from nan_checker import nan_checker

df_nan = nan_checker(df)
df_nan

df_miss =df_nan[df_nan['dtype'] == 'float64'].reset_index(drop=True)
df_miss
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
# %%

'''
Now converting categorical columns to ordinal/numeric values.
    gender             |Male|Female --> 1|0
    hypertension       |1|0  
    heart_disease      |1|0  
    ever_married       |Yes|No --> 1|0 
    work_type          |Private|Self-emp.|children|Govt_job|Never_worked --> 0|1|2|3|4 
    Residence_type     |Urban|Rural --> 1|0 
    smoking_status     |never_smoked|unknown|formerly_smoked|smokes --> 0|1|2|3
 
'''
#%%
df['gender'].replace(to_replace=['Male', 'Female'], value=['1', '0'], inplace=True)
df['ever_married'].replace(to_replace=['Yes', 'No'], value=['1', '0'], inplace=True)
df['work_type'].replace(to_replace=['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'], value=['0', '1', '2', '3', '4'], inplace=True)
df['Residence_type'].replace(to_replace=['Urban', 'Rural'], value=['1', '0'], inplace=True)
df['smoking_status'].replace(to_replace=['never smoked', 'Unknown', 'formerly smoked', 'smokes'], value=['0', '1', '2', '3'], inplace=True)

df.head()

#%%
# BMI column has 201 NaN values. Dropping rows with NaN values. 
# drop all rows that have any NaN values
df = df.dropna()

# reset index of DataFrame
df = df.reset_index(drop=True)

df.shape
#%%
value_count_stroke = df['stroke'].value_counts()
print(value_count_stroke)

axstroke=df['stroke'].value_counts().plot.bar()
plt.title('Stroke Frequency')
plt.xlabel('Class')
axstroke.set_xticklabels(['No Stroke', 'Stroke'])
plt.ylabel('Frequency')
plt.show()

# 0    4699
# 1     209
# Name: stroke, dtype: int64

# From initial analysis, the dataset seems to be highly unbalanced. 
# There are 4699 cases without a stroke and 209 cases with a stroke among the participant list. 
#%%
# Further subdividing the dataset into male and female sets. 

grouped = df.groupby(df['gender'])
male_stroke = grouped.get_group('1')
female_stroke = grouped.get_group('0')
# %%
print(male_stroke['stroke'].value_counts())
# 0    1922
# 1     89
# Name: stroke, dtype: int64
# Extremely unbalanced dataset. 108 men with stroke and 2007 without stroke. 
# See plot below
axmale = male_stroke['stroke'].value_counts().plot.bar()
plt.title('Male Stroke Frequency')
plt.xlabel('Class')
axmale.set_xticklabels(['No Stroke', 'Stroke'])
plt.ylabel('Frequency')
plt.show()

#%%
print(female_stroke['stroke'].value_counts())
# 0    2777
# 1     120
# Name: stroke, dtype: int64
# Extremely unbalanced dataset. 141 women with stroke and 2853 without stroke. 
# See plot below
axfemale=female_stroke['stroke'].value_counts().plot.bar()
plt.title('Female Stroke Frequency')
plt.xlabel('Class')
axfemale.set_xticklabels(['No Stroke', 'Stroke'])
plt.ylabel('Frequency')
plt.show()

# %%
# Above analysis suggests that the data set is highly unbalanced. 
# This will require balancing the dataset (target variable) before any furthur tests or analysis can be conducted! 

# Balancing the data set using SMOTE: Synthetic Minority Oversampling Technique. Using this method, as the name suggests, the minority target variable is oversampled using random values. The technique uses the concept of K-NN or K neareast neighbors to intelligently generate synthetic data which resembles the values or shape of the outnumbered data instead of directly copying or reusing pre-existing values. 
# For more info: https://github.com/scikit-learn-contrib/imbalanced-learn



# separating the target variable from the main data set. 

X = df.drop('stroke', axis = 'columns') # regressor data set
y = df['stroke'] # target variable data set

print(y.value_counts())
# 0    4699
# 1     209
# Name: stroke, dtype: int64


#%%
# Converting column data type to int64 so ordinal values remain as int and not get float values when SMOTE is being performed as the process will generate synthetic values based on KNN algorithm. 
# For eg: We have to make sure that column values stay IN [1,0] and not something like 0.55 when synthetic values are being set up. 

X['gender'] = X['gender'].astype(np.int64)
X['ever_married'] = X['ever_married'].astype(np.int64)
X['Residence_type'] = X['Residence_type'].astype(np.int64)
X['work_type'] = X['work_type'].astype(np.int64)
X['smoking_status'] = X['smoking_status'].astype(np.int64)

X.info()


#%%
# EDA - Box plots to show relationships between stroke and work-type & stroke and marriage status 
work_ranking = ["0", "1", "2", "3", "4"]
ax = sns.boxplot(x="work_type", y="bmi", color="b", order=work_ranking, data=df)
plt.title('BMI Distribution based on Work Type')
ax.set_xticklabels(['Private', 'Self-emp.', 'Children', 'Gov Job', 'Never Worked'])
plt.show()
print("\nReady to continue.")

ax1 = sns.boxplot(x="ever_married", y="bmi", color="b", data=df)
plt.title('BMI Distribution based on Marital Status')
ax1.set_xticklabels(['Yes', 'No'])
plt.show()

print("\nReady to continue.")

ax2 = sns.boxplot(x="Residence_type", y="avg_glucose_level", color="b", data=df)
plt.title('Glucose Level Distribution based on Residence Type')
ax2.set_xticklabels(['Urban', 'Rural'])
plt.show()

print("\nReady to continue.")

ax3 = sns.boxplot(x="Residence_type", y="bmi", color="b", data=df)
plt.title('BMI Distribution based on Residence Type')
ax3.set_xticklabels(['Urban', 'Rural'])
plt.show()

print("\nReady to continue.")


#%%
# EDA - Stacked bar charts to visualize stroke  hypertension and heart disease 
pivot_heart = pd.pivot_table(data=stroke_yes, values='stroke', index='heart_disease', columns='gender', aggfunc='count')
ax5 = pivot_heart.plot.bar(stacked=True)
ax5.set_title('Count of Stroke Victims with Heart Disease')
ax5.set_xticklabels(['No', 'Yes'])
print(pivot_heart)
pivot_hyper = pd.pivot_table(data=stroke_yes, values='stroke', index='hypertension', columns='gender', aggfunc='count')
ax6 = pivot_hyper.plot.bar(stacked=True)
ax6.set_title('Count of Stroke Victims with Hypertension')
ax6.set_xticklabels(['No', 'Yes'])
print(pivot_hyper)
#%%
# install imbalanced-learn package that has SMOTE. 
# pip install imbalanced-learn --user
import imblearn
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy = 'minority')
X_sm, y_sm = smote.fit_resample(X, y)
# %%

print(y_sm.value_counts())

# 1    4699
# 0    4699
# Name: stroke, dtype: int64

# We now have generated equal number of participants who have a stroke and participants who do not have a stroke. 

# The data set is perfectly balanced now!

#%%
from sklearn.model_selection import train_test_split
# Logistic regression using balanced dataset

df_unbalanced_train, df_unbalanced_test, y_unbalanced_train, y_unbalanced_test = train_test_split(df, y, test_size= 0.2, random_state= 15, stratify=y)

import statsmodels.api as sm
from statsmodels.formula.api import glm

model1_unbalanced = glm(formula='stroke ~ C(gender) + age + C(hypertension) + C(heart_disease) + C(ever_married) + C(work_type) + C(Residence_type) + avg_glucose_level + bmi + C(smoking_status)', data=df_unbalanced_train, family=sm.families.Binomial())

model1_unbalanced_fit = model1_unbalanced.fit()
print( model1_unbalanced_fit.summary() )

# %%
model2_unbalanced = glm(formula='stroke ~ age + C(hypertension) + C(heart_disease) + avg_glucose_level', data=df_unbalanced_train, family=sm.families.Binomial())

model2_unbalanced_fit = model2_unbalanced.fit()
print( model2_unbalanced_fit.summary() )

# The p-value are all lower than 0.05, meaning that the variables are significant.

# %%
# And let us check the VIF value (watch out for multicollinearity issues)
# Import functions
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
X_unbalanced_train = df_unbalanced_train[['hypertension', 'heart_disease', 'avg_glucose_level', 'age']]
X_unbalanced_train['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X_unbalanced_train.columns
vif["VIF"] = [ variance_inflation_factor(X_unbalanced_train.values, i) for i in range(X_unbalanced_train.shape[1]) ] # list comprehension

# View results using print
print(vif)
# It seems that the vif is lower than 10, which means the model have no problem of Multicollinearity

#%%
# So we can predict the stroke possibility now.

df_train_with_predict = df_unbalanced_train.copy()
df_train_with_predict['predict'] = model2_unbalanced_fit.predict(X_unbalanced_train)

#%%
stroke_lst = [0, 1]

for status in stroke_lst:
    
    subset = df_train_with_predict[df_train_with_predict['stroke'] == status]
    
    # Draw the density plot
    sn.distplot(subset['predict'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = status)
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'Stroke')
plt.title('Density Plot with Different Stroke Status')
plt.xlabel('Prediction')
plt.ylabel('Density')

# %%

# probs_y is a 2-D array of probability of being labeled as 0 (first column of array) vs 1 (2nd column in array)

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
precision, recall, thresholds = precision_recall_curve(df_train_with_predict['stroke'], df_train_with_predict['predict']) 
   #retrieve probability of being 1(in second column of probs_y)
pr_auc = auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

#%%
# from sklearn.metrics import ConfusionMatrixDisplay

# cutoff = 0.5
# ConfusionMatrixDisplay.from_predictions(df_train_with_predict['stroke'], df_train_with_predict['predict']>cutoff)
# We can find that cutoff 0.5 is not suitable for this model.

#%%
# Choosing the Suitable Cutoff Value
cost_fp = 1
cost_fn = 6
# Among the 97,374 hospitalizations (average cost: $20,396 ± $23,256), the number with ischemic, hemorrhagic, or other strokes was 62,637, 16,331, and 48,208, respectively, with these types having average costs, in turn, of $18,963 ± $21,454, $32,035 ± $32,046, and $19,248 ± $21,703. 
# https://pubmed.ncbi.nlm.nih.gov/23954598/#:~:text=Results%3A%20Among%20the%2097%2C374%20hospitalizations,%2432%2C046%2C%20and%20%2419%2C248%20%C2%B1%20%2421%2C703.

cost_lst = []
cutoff_lst = []
for cutoff in np.linspace(0,0.5,26):
    matrix = confusion_matrix(df_train_with_predict['stroke'], df_train_with_predict['predict']>cutoff)
    fp = matrix[0][1]
    fn = matrix[1][0]
    cost_lst.append(fp * cost_fp + fn * cost_fn)
    cutoff_lst.append(cutoff)

plt.title("Cost vs Threshold Chart")
plt.plot(cutoff_lst, cost_lst, "b--")
plt.ylabel("Cost")
plt.xlabel("Threshold")
plt.show()

plt.title("Cost vs Threshold Chart")
plt.plot(cutoff_lst, cost_lst, "b--")
plt.ylabel("Cost")
plt.xlabel("Threshold")
# plt.ylim([750,1200])
plt.ylim([800,1500])
plt.show()

#%%
def cal_precision_recall(confusion_matric):
    TN, FP = confusion_matric[0]
    FN, TP = confusion_matric[1]
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return Precision, Recall

matrix_unbalanced_train = confusion_matrix(df_train_with_predict['stroke'], df_train_with_predict['predict']>0.1)

precision_un_logis, recall_un_logis = cal_precision_recall(matrix_unbalanced_train)

print(classification_report(df_train_with_predict['stroke'], df_train_with_predict['predict']>0.1))

#%% 
# The data before are train data. We use test data here this time to evaluate model again.
X_unbalanced_test = df_unbalanced_test[['hypertension', 'heart_disease', 'avg_glucose_level', 'age']]
X_unbalanced_test['Intercept'] = 1

df_test_with_predict = df_unbalanced_test.copy()
df_test_with_predict['predict'] = model2_unbalanced_fit.predict(X_unbalanced_test)

matrix_unbalanced_test = confusion_matrix(df_test_with_predict['stroke'], df_test_with_predict['predict']>0.1)
print(matrix_unbalanced_test)
print(classification_report(df_test_with_predict['stroke'], df_test_with_predict['predict']>0.1))

#%%
from sklearn.metrics import roc_auc_score, roc_curve
roc_auc_score(df_test_with_predict['stroke'], df_test_with_predict['predict']>0.1)

# from sklearn.metrics import ConfusionMatrixDisplay
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=matrix_unbalanced_train,
# )
# disp.plot()
# plt.show()


#%%
# KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report

X_scale = pd.DataFrame( scale(X_sm), columns=X_sm.columns )
y_scale = y_sm.copy()

X_scale_train, X_scale_test, y_scale_train, y_scale_test = train_test_split(X_scale, y_scale, test_size= 0.2, random_state= 15, stratify=y_scale)

#%%
mrroger_lst = []
train_score_lst = []
test_score_lst = []

for mrroger in range(1,26):
    knn = KNeighborsClassifier(n_neighbors=mrroger)
    knn.fit(X_scale_train, y_scale_train)

    # y_scale_test_pred = knn.predict(X_scale_test)

    mrroger_lst.append(mrroger)
    train_score_lst.append( knn.score(X_scale_train,y_scale_train) )
    test_score_lst.append( knn.score(X_scale_test,y_scale_test) )

#%%

plt.title("KNN: Varying Number of Neighbors")
plt.plot(mrroger_lst, train_score_lst, "b--", label="Train")
plt.plot(mrroger_lst, test_score_lst, "r--", label="Test")
plt.ylabel("Accuracy")
plt.xlabel("Number of Neighbors")
plt.legend(loc="lower left")


# So choosing k as 7
#%%
mrroger = 7
# whether scale or no scale
# no scale
X_no_scale_train, X_no_scale_test, y_no_scale_train, y_no_scale_test = train_test_split(X_sm, y_sm, test_size= 0.2, random_state= 15, stratify=y_sm)

knn_no_scale = KNeighborsClassifier(n_neighbors=mrroger)
knn_no_scale.fit(X_no_scale_train, y_no_scale_train)

# y_no_scale_train_pred = knn_no_scale.predict(X_no_scale_train)

# print(classification_report(y_no_scale_train, y_no_scale_train_pred))

# knn_no_scale_confusion_matric = confusion_matrix(y_no_scale_train, y_no_scale_train_pred)
# print(knn_no_scale_confusion_matric)

# Precision_no_scale, Recall_no_scale = cal_precision_recall(knn_no_scale_confusion_matric)

y_no_scale_test_pred = knn_no_scale.predict(X_no_scale_test)

print(classification_report(y_no_scale_test, y_no_scale_test_pred))

knn_no_scale_confusion_matric = confusion_matrix(y_no_scale_test, y_no_scale_test_pred)
print(knn_no_scale_confusion_matric)

Precision_no_scale, Recall_no_scale = cal_precision_recall(knn_no_scale_confusion_matric)

#%%
# scaled

knn_scale = KNeighborsClassifier(n_neighbors=mrroger)
knn_scale.fit(X_scale_train, y_scale_train)


# y_scale_train_pred = knn_scale.predict(X_scale_train)

# print(classification_report(y_scale_train, y_scale_train_pred))

# knn_scale_confusion_matric = confusion_matrix(y_scale_train, y_scale_train_pred)
# print(knn_scale_confusion_matric)

# Precision_scale, Recall_scale = cal_precision_recall(knn_scale_confusion_matric)

y_scale_test_pred = knn_scale.predict(X_scale_test)

print(classification_report(y_scale_test, y_scale_test_pred))

knn_scale_confusion_matric = confusion_matrix(y_scale_test, y_scale_test_pred)
print(knn_scale_confusion_matric)

print(roc_auc_score(y_scale_test, y_scale_test_pred))

Precision_scale, Recall_scale = cal_precision_recall(knn_scale_confusion_matric)
#%%

df_knn = pd.DataFrame(data={'index': ['Precision', 'Recall'], 'No scale': [Precision_no_scale,  Recall_no_scale ], 'Scale': [Precision_scale, Recall_scale]})

df_knn.set_index('index', inplace=True)

df_knn[['No scale','Scale']].plot(kind='bar', rot=0, title='Precision and Recall Value for No Scale and Scale KNN').legend(loc='lower right')
plt.xlabel("")
plt.show()

#%%
# from sklearn.metrics import ConfusionMatrixDisplay
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=knn_scale_confusion_matric,
#     display_labels=knn_scale.classes_
# )
# disp.plot()
# plt.show()

# %%
# Logistic regression using sklearn:
#
# Creating test/train data sets from the balanced set using sklearn train_test_split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)

print(y_train.value_counts()) # 80% as training set   
print(y_test.value_counts()) # 20% as test set

# %%
from sklearn.linear_model import LogisticRegression

strokemodel = LogisticRegression(max_iter=1000)

strokemodel.fit(X_train, y_train)

#%%
strokemodel.score(X_test, y_test)
# 0.81915 The model is nearly 82% effective in identifying or predicting the results of a participant getting a stroke or not. 
# %%
# creating predicted target values using the model for X_test:
y_predict = strokemodel.predict(X_test)

# Now we can compare y_predict(predicted) values with actual y_test(real) values using a confusion matrix:

cm_stroke_model = confusion_matrix(y_test, y_predict)
# array([[751, 189],
#        [151, 789]], dtype=int64)
#%%

# Creating a heatmap of the above confusion matrix for better visualization and understanding:

sn.heatmap(cm_stroke_model, annot = True, fmt="d")
plt.show()

# y axis - truth values
# x axis - predicted values

# The heatmap of the confusion matrix shows that when the values in the y_test data set are compared with the predicted values from y_predict.
# 751 participants were predicted to 'not have a stroke' and 751 time it was predicted right by the model. 189 times the participant 'did not have a stroke' but it was predicted they did. 151 participants were predidcted to 'have a stroke' but in reality they did not suffer from one. 789 times the participants were predicted to 'have a stroke' and 789 times they got one. 

print(classification_report(y_test, y_predict))

#              precision    recall  f1-score   support
#
#           0       0.83      0.80      0.82       940
#           1       0.81      0.84      0.82       940
#
#    accuracy                           0.82      1880
#   macro avg       0.82      0.82      0.82      1880
# weighted avg       0.82      0.82      0.82      1880

# Accuracy and f1-score of the model is 0.82 or 82% which is pretty decent given that the target variable has been modified.


# %%
#feature selection for the stroke model. 
from sklearn.feature_selection import RFE

selector = RFE(strokemodel, n_features_to_select=6, step=1)
selector = selector.fit(X_train, y_train)
print(selector.support_)
print(selector.ranking_)

# [ True False  True  True  True  True  True False False False]
# [1 2 1 1 1 1 1 5 4 3]
#  residence type, work type, married status, heart disease, hypertension, and gender have more variance on the target variable than other variables. 

#%%
# We will check for multicollinearity between the variables 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
X_train_fs = X_train[['hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'gender']]


# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X_train_fs.columns
vif["VIF"] = [ variance_inflation_factor(X_train_fs.values, i) for i in range(X_train_fs.shape[1]) ] # list comprehension

# View results using print
print(vif)
# It seems that the vif is lower than 10, which means the model has no problem of Multicollinearity.


# %%
# Generating a secong Logit model based off on features selected from the above analysis:

# Logit 2 - Feature Selected Model

X1_train, X1_test, y1_train, y1_test   = train_test_split(
    X_train_fs, y_train, test_size=0.2, random_state=15, stratify=y_train)

strokemodel2 = LogisticRegression(max_iter=1000)

strokemodel2.fit(X_train_fs, y_train)
# 65% Accuracy
#%%
strokemodel2.score(X1_test, y1_test)
# %%
y_pred_model2 = strokemodel2.predict(X1_test)
# Now we can compare y_predict(predicted) values with actual y_test(real) values using a confusion matrix:

cm_stroke_model2 = confusion_matrix(y1_test, y_pred_model2)
# array([[437, 315],
#        [198, 554]], dtype=int64)


print(classification_report(y1_test, y_pred_model2))

#               precision    recall  f1-score   support

#            0       0.69      0.58      0.63       752
#            1       0.64      0.74      0.68       752

#     accuracy                           0.66      1504
#    macro avg       0.66      0.66      0.66      1504
# weighted avg       0.66      0.66      0.66      1504
# %%

# Creating a heatmap of the above confusion matrix for better visualization and understanding:

sn.heatmap(cm_stroke_model2, annot=True, fmt="d")
plt.show()


#%%
# Classification tree model on the same data to compare with the Logit model. 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# %%
tree_model = DecisionTreeClassifier(max_depth=3, random_state=1)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
tree_cm = confusion_matrix(y_test, y_pred_tree)
print(tree_cm)
sn.heatmap(tree_cm, annot = True, fmt = 'd')
plt.show()
print(classification_report(y_test, y_pred_tree))
#
# As max_depth is increased in the tree, the accuracy level of the model gets slightly better each time. 
# This suggests that increasing the depth tends to overfit the model, which makes sense as increasing the tree depth to a value high enough will essentially lead to each data point in the stroke set being a leaf node by itself. 
#   0       1
# 0 [[615 325]
# 1 [ 63 877]]

# 615 participants were predicted to not have a stroke and 615 participants were predicted correctly.
# 63 participants had a stroke but were predicted incorrectly by the model to not have a stroke. 
# 325 participants did not have a stroke but were predicted incorrectly by the model to have a stroke. 
# 877 participants had a stroke and were predicted correctly by the model to have a stroke. 

#               precision    recall  f1-score   support
# 
#            0       0.91      0.65      0.76       940
#            1       0.73      0.93      0.82       940
# 
#     accuracy                           0.79      1880
#    macro avg       0.82      0.79      0.79      1880
# weighted avg       0.82      0.79      0.79      1880

# Overall, accuracy is 79%, which is slightly less than the Logit model. 
#%%
# Tree structure using graphviz library:

from sklearn.tree import export_graphviz 

dot_data = export_graphviz(tree_model, out_file = 'tree1.dot', 
                                feature_names =['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
#%%
# Running cross-validation on all the models. 
logit_cv = cross_val_score(strokemodel, X_train, y_train, cv= 10, scoring='accuracy')
print(logit_cv)

dtc_cv = cross_val_score(tree_model, X_train, y_train, cv= 10, scoring='accuracy')
print(dtc_cv)

logit_fs = cross_val_score(strokemodel2, X_train_fs, y_train, cv= 10, scoring='accuracy')
print(logit_fs)

#%%
# Generating the ROC and AUC plots for both the models:
from sklearn.metrics import roc_auc_score, roc_curve

# ROC/AUC for Logit model: 
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_predict)
plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'r')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightsalmon', alpha=0.6)
plt.text(0.95, 0.05, 'AUC = %0.4f' % roc_auc_score(y_test, y_predict), ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# AUC = 0.8207

# ROC/AUC for Classification Tree model:
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred_tree)
plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='steelblue', alpha=0.6)
plt.text(0.95, 0.05, 'AUC = %0.4f' % roc_auc_score(y_test, y_pred_tree), ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# AUC = 0.7926
# Both models are either over or very close to the 0.80 AUC mark. 
# %%
