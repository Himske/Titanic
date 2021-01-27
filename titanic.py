import warnings

# preprocessing and machine learning
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv

# python imports
import math
import time
import random
import datetime

# data manipulation
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

warnings.filterwarnings('ignore')

FOLDS = 10


def plot_count_dist(data, label_column, target_column, figsize=(20, 5)):
    '''Function to plot counts and distribution of a label variable and target variable side by side.'''
    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    sns.countplot(y=target_column, data=data)
    plt.subplot(1, 2, 2)
    sns.distplot(data.loc[data[label_column] == 1][target_column], kde_kws={'label': 'Survived'})
    sns.distplot(data.loc[data[label_column] == 0][target_column], kde_kws={'label': 'Did not survive'})


def transform_data(data, dropna_embarked=False):
    df_transformed = data.copy()

    if dropna_embarked:
        df_transformed = df_transformed.dropna(subset=['Embarked'])
    else:
        df_transformed.Embarked = df_transformed.Embarked.fillna('S')

    df_transformed['Title'] = df_transformed.Name.str.split(',').str.get(1).str.split('.').str.get(0)

    # fill missing age values based on mean per title, not sure if this is the best method
    title_age_dict = df_transformed.groupby('Title').Age.mean().to_dict()
    df_transformed.Age = df_transformed.Age.fillna(df_transformed.Title.map(title_age_dict))
    # fill by average
    # df_transformed.Age = df_transformed.Age.fillna(df_transformed.Age.dropna().mean())

    df_transformed['AgeB'] = pd.cut(df_transformed.Age, bins=9)
    labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
    df_transformed['Age_Bins'] = pd.cut(df_transformed.Age,
                                        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                                        labels=labels)

    df_transformed['FareB'] = pd.cut(df_transformed.Fare, bins=14)
    labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70',
              '70-80', '80-90', '90-100', '100-150', '150-200', '200-300', '300-600']
    df_transformed['Fare_Bins'] = pd.cut(df_transformed.Fare,
                                         bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 600],
                                         labels=labels,
                                         include_lowest=True)

    return df_transformed


def encode_data(data):
    df_encoded = data.copy()

    df_encoded = pd.get_dummies(df_encoded, columns=['Pclass', 'Sex', 'Embarked'])
    df_encoded.rename(columns={'Sex_female': 'Female', 'Sex_male': 'Male'}, inplace=True)

    return df_encoded


def examine_data(data):
    print('Sample')
    print(data.head(), end='\n\n')
    print('Describe')
    print(data.describe(), end='\n\n')

    # are there missing values?
    missingno.matrix(data, figsize=(30, 10))
    print('Total null values per column')
    print(data.isnull().sum(), end='\n\n')
    print('Percentage of null values per column')
    print(round(data.isnull().sum() / len(data) * 100, 2), end='\n\n')

    # Pclass = Ticket class. Split Pclass into binary columns.
    plot_count_dist(data, label_column='Survived', target_column='Pclass')

    # Age
    print('Age')
    print(data.Age.value_counts(), end='\n\n')
    plot_count_dist(data, label_column='Survived', target_column='Age')

    # SibSp = # of siblings / spouses aboard the Titanic
    print('SibSp')
    print(data.SibSp.value_counts(), end='\n\n')
    plot_count_dist(data, label_column='Survived', target_column='SibSp')

    # Parch = # of parents / children aboard the Titanic
    print('Parch')
    print(data.Parch.value_counts(), end='\n\n')
    plot_count_dist(data, label_column='Survived', target_column='Parch')

    # Ticket
    print('Ticket')
    print(data.Ticket.value_counts(), end='\n\n')

    # Fare
    print('Fare')
    print(data.Fare.value_counts(), end='\n\n')

    # Cabin
    print('Cabin')
    print(data.Cabin.value_counts(), end='\n\n')

    # Embarked, C = Cherbourg, Q = Queenstown, S = Southampton
    print('Embarked')
    print(data.Embarked.value_counts(), end='\n\n')


def rule_based_guess(data, cv):
    '''
    rich women survived, men in 2nd and 3rd class did not survive\n
    rich men and women in 2nd and 3rd class about 50% chance
    '''
    start_time = time.time()

    df_rule = pd.DataFrame()
    df_rule['Survived'] = data.Survived

    # One pass
    df_rule['Hyp'] = 0

    df_rule.loc[data.Female == 1, 'Hyp'] = np.random.randint(0, 2, size=(len(data[data.Female == 1]), 1))
    df_rule.loc[(data.Female == 1) & (data.Pclass_1 == 1), 'Hyp'] = 1
    df_rule.loc[(data.Male == 1) & (data.Pclass_1 == 1), 'Hyp'] = \
        np.random.randint(0, 2, size=(len(data[(data.Male == 1) & (data.Pclass_1 == 1)]), 1))

    df_rule['Rule_Guess'] = 0
    df_rule.loc[df_rule.Survived == df_rule.Hyp, 'Rule_Guess'] = 1
    acc = round(df_rule.Rule_Guess.value_counts(normalize=True)[1] * 100, 2)

    train_pred = df_rule.Rule_Guess.values

    # Cross validation
    cv_scores = list()

    for _ in range(cv):
        df_rule['Hyp'] = 0

        df_rule.loc[data.Female == 1, 'Hyp'] = np.random.randint(0, 2, size=(len(data[data.Female == 1]), 1))
        df_rule.loc[(data.Female == 1) & (data.Pclass_1 == 1), 'Hyp'] = 1
        df_rule.loc[(data.Male == 1) & (data.Pclass_1 == 1), 'Hyp'] = \
            np.random.randint(0, 2, size=(len(data[(data.Male == 1) & (data.Pclass_1 == 1)]), 1))

        df_rule['Rule_Guess'] = 0
        df_rule.loc[df_rule.Survived == df_rule.Hyp, 'Rule_Guess'] = 1

        cv_scores.append(df_rule.Rule_Guess.value_counts(normalize=True)[1])

    execution_time = time.time() - start_time

    return train_pred, acc, round(np.mean(cv_scores) * 100, 2), execution_time


def fit_ml_algo(title, algo, X_train, y_train, cv):
    '''Function that runs the requested algorithm and returns the accuracy metrics'''
    start_time = time.time()

    # One pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)

    # Cross validation
    train_pred = model_selection.cross_val_predict(estimator=algo, X=X_train, y=y_train, cv=cv, n_jobs=-1)

    # Cross validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    execution_time = time.time() - start_time

    add_score(models, title, acc)
    add_score(models_cv, title, acc_cv)
    print(f'{title} execution time: {execution_time}')

    return train_pred


def feature_importance(model, data):
    '''Function to show which feature is most important in the model'''
    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=True).iloc[-30:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 20))
    return fea_imp


# load the data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_gender_submission = pd.read_csv('data/gender_submission.csv')

# examine training data
# examine_data(df_train)

# transform data
df_transformed = transform_data(df_train)

# encode data
df_encoded = encode_data(df_transformed)
df_encoded.to_csv('data/train_transformed.csv')

# Create X_train and y_train to use in the sklearn models
df_selected = df_encoded[['Female', 'Male', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Fare_Bins', 'Age_Bins', 'SibSp',
                          'Parch', 'Title', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

X_train = df_selected.apply(LabelEncoder().fit_transform)

df_selected_nobin = df_encoded[['Female', 'Male', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Fare', 'Age', 'SibSp',
                                'Parch', 'Title', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

X_train_nobin = df_selected_nobin.apply(LabelEncoder().fit_transform)

y_train = df_encoded.Survived

models = {'Model': [], 'Score': []}
models_cv = {'Model': [], 'Score': []}


def add_score(in_dict, title, score):
    in_dict['Model'].append(title)
    in_dict['Score'].append(score)


# Rule based guess
train_pred_rule, acc_rule, acc_cv_rule, execution_time = rule_based_guess(df_encoded, FOLDS)
title = 'Rule based'
print(title)
add_score(models, title, acc_rule)
add_score(models_cv, title, acc_cv_rule)
print(f'{title} execution time: {execution_time}')

# Logistic regression
title = 'Logistic regression'
print(title)
train_pred_log = fit_ml_algo(title, LogisticRegression(max_iter=1000), X_train, y_train, FOLDS)

# Logistic regression without the bin features
title = 'Logistic regression no bins'
print(title)
train_pred_log_nobin = fit_ml_algo(title, LogisticRegression(max_iter=1000), X_train_nobin, y_train, FOLDS)

# K-Nearest Neighbor
title = 'K-Nearest Neighbor'
print(title)
train_pred_knn = fit_ml_algo(title, KNeighborsClassifier(), X_train, y_train, FOLDS)

# K-Nearest Neighbor without the bin features
title = 'K-Nearest Neighbor no bins'
print(title)
train_pred_knn_nobin = fit_ml_algo(title, KNeighborsClassifier(), X_train_nobin, y_train, FOLDS)

# Gaussian Naive Bayes
title = 'Gaussian Naive Bayes'
print(title)
train_pred_gaussian = fit_ml_algo(title, GaussianNB(), X_train, y_train, FOLDS)

# Gaussian Naive Bayes without bin feature
title = 'Gaussian Naive Bayes no bins'
print(title)
train_pred_gaussian_nobin = fit_ml_algo(title, GaussianNB(), X_train_nobin, y_train, FOLDS)

# Linear Support Vector Machines (SVC)
title = 'Linear Support Vector Machines (SVC)'
print(title)
train_pred_svc = fit_ml_algo(title, LinearSVC(max_iter=100000), X_train, y_train, FOLDS)

# Linear Support Vector Machines (SVC) without bin feature
title = 'Linear Support Vector Machines (SVC) no bins'
print(title)
train_pred_svc_nobin = fit_ml_algo(title, LinearSVC(max_iter=10000000), X_train_nobin, y_train, FOLDS)

# Stochastic Gradient Descent
title = 'Stochastic Gradient Descent'
print(title)
train_pred_sgd = fit_ml_algo(title, SGDClassifier(), X_train, y_train, FOLDS)

# Stochastic Gradient Descent without bin feature
title = 'Stochastic Gradient Descent no bins'
print(title)
train_pred_sgd_nobin = fit_ml_algo(title, SGDClassifier(), X_train_nobin, y_train, FOLDS)

# Descision Tree Classifier
title = 'Descision Tree Classifier'
print(title)
train_pred_dt = fit_ml_algo(title, DecisionTreeClassifier(), X_train, y_train, FOLDS)

# Descision Tree Classifier without bin feature
title = 'Descision Tree Classifier no bins'
print(title)
train_pred_dt_nobin = fit_ml_algo(title, DecisionTreeClassifier(), X_train_nobin, y_train, FOLDS)

# Gradient Boost Trees
title = 'Gradient Boost Trees'
print(title)
train_pred_gbt = fit_ml_algo(title, GradientBoostingClassifier(), X_train, y_train, FOLDS)

# Gradient Boost Trees without bin feature
title = 'Gradient Boost Trees no bins'
print(title)
train_pred_gbt_nobin = fit_ml_algo(title, GradientBoostingClassifier(), X_train_nobin, y_train, FOLDS)

# Catboost
title = 'Catboost'
print(title)
cat_features = np.where(X_train.dtypes != np.float)[0]
train_pool = Pool(X_train, y_train, cat_features)

catboost_model = CatBoostClassifier(iterations=1000, custom_loss=['Accuracy'], loss_function='Logloss')
catboost_model.fit(train_pool, plot=False, verbose=False)

acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)

start_time = time.time()
cv_params = catboost_model.get_params()
cv_data = cv(train_pool, cv_params, fold_count=FOLDS, plot=False, verbose=False)
execution_time = time.time() - start_time
acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)

add_score(models, title, acc_catboost)
add_score(models_cv, title, acc_cv_catboost)
print(title, execution_time)

# Catboost without bin feature
title = 'Catboost no bins'
print(title)
cat_features_nobin = np.where(X_train_nobin.dtypes != np.float)[0]
train_pool_nobin = Pool(X_train_nobin, y_train, cat_features_nobin)

catboost_model.fit(train_pool_nobin, plot=False, verbose=False)

acc_catboost_nobin = round(catboost_model.score(X_train_nobin, y_train) * 100, 2)

start_time = time.time()
cv_params = catboost_model.get_params()
cv_data = cv(train_pool_nobin, cv_params, fold_count=FOLDS, plot=False, verbose=False)
execution_time = time.time() - start_time
acc_cv_catboost_nobin = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)

add_score(models, title, acc_catboost_nobin)
add_score(models_cv, title, acc_cv_catboost_nobin)
print(f'{title} execution time: {execution_time}')

print('Accuracy scores')
print(pd.DataFrame(models).sort_values(by='Score', ascending=False), end='\n\n')

print('Cross validation accuracy scores')
print(pd.DataFrame(models_cv).sort_values(by='Score', ascending=False))

feature_importance(catboost_model, X_train_nobin)

# Precision and Recall
metrics = ['Precision', 'Recall', 'F1', 'AUC']
eval_metrics = catboost_model.eval_metrics(train_pool_nobin, metrics=metrics, plot=False)

for metric in metrics:
    print(f'{metric}: {np.mean(eval_metrics[metric])}')

# Make predictions on the test set using selected model (Catboost with no bins)
df_test_transformed = transform_data(df_test)
df_test_encoded = encode_data(df_test_transformed)

df_test_selected = df_test_encoded[['Female', 'Male', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Fare', 'Age', 'SibSp',
                                    'Parch', 'Title', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

X_test = df_test_selected.apply(LabelEncoder().fit_transform)

predictions = catboost_model.predict(X_test)

# Create submission file
submission = pd.DataFrame()
submission['PassengerId'] = df_test.PassengerId
submission['Survived'] = predictions
submission.Survived = submission.Survived.astype(int)

submission.to_csv('data/submission.csv', index=False)
