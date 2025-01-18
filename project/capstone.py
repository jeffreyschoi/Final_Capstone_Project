import random
import numpy as np
import pandas as pd

random.seed(17226773)

data = pd.read_csv('spotify52kData.csv')

## Seperating numerical and categorical data
numericals = data.select_dtypes(include=['int64','float64']).columns
categoricals = data.select_dtypes(include=['bool','object']).columns

## Fill missing values with mean
data[numericals] = data[numericals].fillna(data[numericals].median())

## Fill missing values with mode
for i in categoricals:
    mode = data[i].mode()[0]
    data[i] = data[i].fillna(mode)

## Question 1
print("Question 1:")
import matplotlib.pyplot as plt

Q1features = data[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

## Plot distribution for each feature
for i, column in enumerate(Q1features.columns):
    axes[i].hist(Q1features[column], bins="auto")
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

from scipy.stats import kstest

## KS Test to see how similar it is to a normal distibution
for i, column in enumerate(Q1features.columns):
    normalDistTest = kstest(Q1features[column].to_numpy().flatten(), "norm")
    print(f"KS test p-value for {column} distribution: ", normalDistTest.pvalue)
    
## Question 2
from scipy.stats import spearmanr
print("\nQuestion 2:")

Q2features = data[['duration', 'popularity']]

## Plot the features to visualize correlation
plt.scatter(Q2features['duration'], Q2features['popularity'].astype(float))
plt.xlabel('duration')
plt.ylabel('populatiry')
plt.title('Song Duration by Popularity')
plt.show()

## r and rho coefficients
print("Pearson's r: ", Q2features.corr().loc['duration', 'popularity'])
print("Spearman's rho: ", spearmanr(Q2features['duration'], Q2features['popularity']).statistic)

## Question 3
print("\nQuestion 3:")
from scipy.stats import mannwhitneyu

Q3features = data[['explicit', 'popularity']]

## Split according to explicit or not
explicitSongs = Q3features.loc[Q3features['explicit'], 'popularity']
notexplicitSongs = Q3features.loc[~Q3features['explicit'], 'popularity']

## MW Test to see if they come from same population
u1,p1 = mannwhitneyu(explicitSongs, notexplicitSongs)
print("p-value: ", p1)

## Print each of their medians
print("Explicit Songs Median:", explicitSongs.median())
print("Not explicit Songs Median:", notexplicitSongs.median())

## Question 4
print("\nQuestion 4:")

Q4features = data[['mode', 'popularity']]

## Split according to major or minor
majorKeySongs = Q4features.loc[Q4features['mode'] == 1, 'popularity']
minorKeySongs = Q4features.loc[Q4features['mode'] == 0, 'popularity']

## MW Test again
u2,p2 = mannwhitneyu(majorKeySongs, minorKeySongs, alternative='greater')
print("p-value: ", p2)

## Print each of their medians
print("Major Key Songs Median:", majorKeySongs.median())
print("Minor Key Songs Median:", minorKeySongs.median())

## Question 5
print("\nQuestion 5:")

Q5features = data[['energy', 'loudness']]

## Plot features to visualize correlation
plt.scatter(Q5features['energy'], Q5features['loudness'])
plt.xlabel('energy')
plt.ylabel('loudness')
plt.title('Song Energy by Loudness')
plt.show()

## r and rho coefficients
print("Pearson's rho: ", Q5features.corr().loc['energy', 'loudness'])
print("Spearman's rho: ", spearmanr(Q5features['energy'], Q5features['loudness']).statistic)

## Question 6
from sklearn.linear_model import LinearRegression
print('\nQuestion 6')

Q6features = data[['popularity']]

## OLS for each feature
best_rSqr = 0
for i, column in enumerate(Q1features.columns):
    x = Q1features[column].to_numpy().reshape(len(data),1) 
    y = Q6features['popularity'] 
    singleFactorModel = LinearRegression().fit(x,y)
    rSqr = singleFactorModel.score(x,y)
    print(f'r-Squared predicted on {column}: ', rSqr)
    if rSqr > best_rSqr:            ## Used to figure out best r-squared value out of all the models
        best_rSqr = rSqr
        best_feature = column   
print(f'\nBest Feature: {best_feature} with R-Squared: ', best_rSqr)

## Question 7
print('\nQuestion 7')

Q7features = data[['popularity']]

## Multiple Linear Regression using all features
multipleFactorModel = LinearRegression().fit(Q1features,Q7features)
print('r_squared: ', multipleFactorModel.score(Q1features,Q7features))

## Question 8
from scipy import stats
from sklearn.decomposition import PCA
print('\nQuestion 8')

Q8featuresZScored = stats.zscore(Q1features) ## Standardize data
Q8featuresPCA = PCA().fit(Q8featuresZScored) ## PCA model

Q8featuresEigVals = Q8featuresPCA.explained_variance_ ## explained varience or eigenvalues
Q8featuresLoadings = Q8featuresPCA.components_ ## each new component

## Visualize Screeplot
plt.bar(np.linspace(1,10,10), Q8featuresEigVals)
plt.plot([0,10],[1,1],color='orange')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

## Determined 7 meaningful principle components
meaningfulComponents = 7
explainedVarianceRatio = Q8featuresPCA.explained_variance_ratio_
sumExplainedVariance = sum(explainedVarianceRatio[:meaningfulComponents])
cumulativeVarianceRatio = np.cumsum(explainedVarianceRatio)

## Plot the explained variance and the cumulative to see where 90% is
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explainedVarianceRatio) + 1), explainedVarianceRatio, marker='o', label='Explained Variance Ratio')
plt.plot(range(1, len(cumulativeVarianceRatio) + 1), cumulativeVarianceRatio, marker='x', linestyle='--', label='Cumulative Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.legend()
plt.grid(True)
plt.show()

print("Eivenvalues",Q8featuresEigVals)
print("Explained Variance for the First Three Principal Components:", sumExplainedVariance)

## Question 9
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
print('\nQuestion 9')

Q9features = data[['mode','valence','acousticness']]
## Split 80% to training and 20% to testing 
## First is valence and mode
X_train, X_test, y_train, y_test = train_test_split(Q9features[['valence']], Q9features['mode'].values, train_size=0.8, random_state=17226773)

model = LogisticRegression()
model.fit(X_train, y_train)

## Calculate AUC value
yPredProbability = model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, yPredProbability)
AUC = auc(fpr, tpr)
print("AUC for valence:", AUC)

## Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

## Split 80% to training and 20% to testing 
## This one is for acousticness and mode
X_train, X_test, y_train, y_test = train_test_split(Q9features[['acousticness']], Q9features['mode'].values, train_size=0.8, random_state=17226773)

model.fit(X_train, y_train)

## Calculate AUC value
yPredProbability = model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, yPredProbability)
AUC = auc(fpr, tpr)
print("AUC for acousticness:", AUC)

## Question 10
print('\nQuestion 10')

Q10features = data[['track_genre','duration']]
## Convert track_genre to a binary  column; 1: classical, 0: not classical
Q10features['classical'] = Q10features['track_genre'].apply(lambda x: 1 if x == 'classical' else 0)

## Split train and test from duration feature and classical feature
X_train, X_test, y_train, y_test = train_test_split(Q10features[['duration']], Q10features['classical'].values, train_size=0.8, random_state=17226773)

model = LogisticRegression()
model.fit(X_train, y_train)

## Calculate AUC curve
yPredProbability = model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, yPredProbability)
AUC = auc(fpr, tpr)
print("AUC for duration:", AUC)

## Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for duration')
plt.legend()
plt.show()

## Extract 7 relecent PCs
releventPCA = PCA(n_components=7).fit_transform(Q8featuresZScored)
## Split train and test from PCs and classical feature
X_train, X_test, y_train, y_test = train_test_split(releventPCA, Q10features['classical'].values, train_size=0.8, random_state=17226773)

model.fit(X_train, y_train)

## Calculate AUC curve
yPredProbability = model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, yPredProbability)
AUC = auc(fpr, tpr)
print("AUC for relevent principle components:", AUC)

## Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for relevent principle components')
plt.legend()
plt.show()

## Extra Credit
print('\nExtra Credit')

ECfeatures = data[['time_signature','danceability','popularity']]

## Determine average danceability rating and counts for each time signature
timesigXdance = ECfeatures.groupby('time_signature')['danceability'].agg(['mean', 'count'])

## Plot average danceability rating for visualization
timesigXpop = ECfeatures.groupby('time_signature')['popularity'].mean()
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Time Signature')
ax1.set_ylabel('Average Danceability Level', color=color)
ax1.bar(timesigXdance.index, timesigXdance['mean'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
## Also plot total counts for each time signature
color = 'tab:red'
ax2.set_ylabel('Count of Values', color=color)
ax2.plot(timesigXdance.index, timesigXdance['count'], color=color, marker='o', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)
plt.title('Average Danceability Level and Count of Values for Each Time Signature')
plt.xlabel('Time Signature')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

## Seperately plot for average popularity rating for visualization
ax1.set_xlabel('Time Signature')
ax1.set_ylabel('Average Danceability Level', color=color)
timesigXpop.plot(kind='bar')
ax1.tick_params(axis='y', labelcolor=color)
plt.title('Average Popularity Level for Each Time Signature')
plt.xlabel('Time Signature')
plt.show()
