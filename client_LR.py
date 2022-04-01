import os
import timeit
import warnings
from collections import defaultdict

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

# Using scikit-learn and flower for federated learning
import flwr as fl
import utils

from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, \
    plot_confusion_matrix
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored

warnings.filterwarnings('ignore')

np.random.seed(100)

# Get the path to dataset
dataset_root = '/home/h4des/Desktop/DoAnChuyenNganh/nsl_kdd_classification/Data/NSL-KDD-Dataset'

train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')

# Data preprocessing
header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type',
                'success_pred']

# Differentiating between nominal, binary, and numeric features

# root_shell is marked as a continuous feature in the kddcup.names
# file, but it is supposed to be a binary feature according to the
# dataset documentation

col_names = np.array(header_names)

nominal_idx = [1, 2, 3]
binary_idx = [6, 11, 13, 14, 20, 21]
numeric_idx = list(set(range(41)).difference(
    nominal_idx).difference(binary_idx))

nominal_cols = col_names[nominal_idx].tolist()
binary_cols = col_names[binary_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()

# tranning attack types
# training_attack_types.txt maps each of the 22 different attacks to 1 of 4 categories

category = defaultdict(list)
category['benign'].append('normal')

with open("/home/h4des/Desktop/DoAnChuyenNganh/nsl_kdd_classification/Data/NSL-KDD-Dataset/training_attack_types.txt", 'r') as f:
    for line in f.readlines():
        attack, cat = line.strip().split(' ')
        category[cat].append(attack)

attack_mapping = dict((v, k) for k in category for v in category[k])

train_df = pd.read_csv(train_file, names=header_names)

train_df['attack_category'] = train_df['attack_type'] \
    .map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)

test_df = pd.read_csv(test_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'] \
    .map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)

train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()

test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()

train_attack_types.plot(kind='barh', figsize=(20, 10), fontsize=20)

train_attack_cats.plot(kind='barh', figsize=(20, 10), fontsize=30)

test_attack_types.plot(kind='barh', figsize=(20, 10), fontsize=15)

test_attack_cats.plot(kind='barh', figsize=(20, 10), fontsize=30)

# Let's take a look at the binary features
# By definition, all of these features should have a min of 0.0 and a max of 1.0
# execute the commands in console

train_df[binary_cols].describe().transpose()

# Wait a minute... the su_attempted column has a max value of 2.0?
train_df.groupby(['su_attempted']).size()

# Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0

train_df['su_attempted'].replace(2, 0, inplace=True)
test_df['su_attempted'].replace(2, 0, inplace=True)
train_df.groupby(['su_attempted']).size()

# Next, we notice that the num_outbound_cmds column only takes on one value!

train_df.groupby(['num_outbound_cmds']).size()

# Now, that's not a very useful feature - let's drop it from the dataset

train_df.drop('num_outbound_cmds', axis=1, inplace=True)
test_df.drop('num_outbound_cmds', axis=1, inplace=True)
numeric_cols.remove('num_outbound_cmds')

"""
Data Preparation

"""

train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category', 'attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category', 'attack_type'], axis=1)

combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(
    combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# use this for catboost
x_train = train_x_raw
x_test = test_x_raw

# Store dummy variable feature names
dummy_variables = list(set(train_x) - set(combined_df_raw))

# execute the commands in console
train_x.describe()
train_x['duration'].describe()

# Experimenting with StandardScaler on the single 'duration' feature
durations = train_x['duration'].values.reshape(-1, 1)
standard_scaler = StandardScaler().fit(durations)
scaled_durations = standard_scaler.transform(durations)
pd.Series(scaled_durations.flatten()).describe()

# Experimenting with MinMaxScaler on the single 'duration' feature

min_max_scaler = MinMaxScaler().fit(durations)
min_max_scaled_durations = min_max_scaler.transform(durations)
pd.Series(min_max_scaled_durations.flatten()).describe()

# Experimenting with RobustScaler on the single 'duration' feature

min_max_scaler = RobustScaler().fit(durations)
robust_scaled_durations = min_max_scaler.transform(durations)
pd.Series(robust_scaled_durations.flatten()).describe()

# Let's proceed with StandardScaler- Apply to all the numeric columns

standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = \
    standard_scaler.transform(train_x[numeric_cols])

test_x[numeric_cols] = \
    standard_scaler.transform(test_x[numeric_cols])

train_x.describe()

train_Y_bin = train_Y.apply(lambda x: 0 if x is 'benign' else 1)
test_Y_bin = test_Y.apply(lambda x: 0 if x is 'benign' else 1)

# logistic regression hyperparameter tuning


# def logistic_reg_grid_search():
#     # Creating a grid of different hyperparameters
#     grid_params = {
#         'penalty': ['l1', 'l2'],
#         'max_iter': [100, 200, 300, 500, 800, 1000]
#     }

#     # logistic regression classifier
#     clf = LogisticRegression(random_state=0)

#     print("Searching for optimal parameters..............")

#     # Building a 10 fold Cross-Validated GridSearchCV object
#     grid_object = GridSearchCV(estimator=clf, param_grid=grid_params, cv=10)

#     print("Training the model...............")

#     # Fitting the grid to the training data
#     grid_object.fit(train_x, train_Y)

#     # Extracting the best parameters
#     print(grid_object.best_params_)

#     # Extracting the best model
#     rf_best = grid_object.best_estimator_
#     print(rf_best)

# Logistic Regression
# def logistic_reg_clf():
#     print("------Logistic Regression Classification-------")

#     # logistic regression classifier
#     clf_lr = LogisticRegression(
#         C=1e5, random_state=0
#     )

#     # start timer
#     starttime = timeit.default_timer()

#     print("Training the Logistic Regression Classifier.......")

#     # train the model
#     clf_lr = clf_lr.fit(train_x, train_Y)

#     print("The time difference is :", timeit.default_timer() - starttime)

#     print("Predicting test data.......")

#     # predict
#     pred_y = clf_lr.predict(test_x)

#     # get results
#     c_matrix = confusion_matrix(test_Y, pred_y)
#     error = zero_one_loss(test_Y, pred_y)
#     score = accuracy_score(test_Y, pred_y)

#     # display results
#     print('Confusion Matrix\n---------------------------\n', c_matrix)
#     print('---------------------------')
#     print("Error: {:.4f}%".format(error * 100))
#     print("Accuracy Score: {:.4f}%".format(score * 100))
#     print(classification_report(test_Y, pred_y))
#     print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

#     # Plot non-normalized confusion matrix
#     disp = plot_confusion_matrix(clf_lr, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
#                                  xticks_rotation='horizontal')
#     plt.title("Confusion Matrix for Logistic Regression")
# plt.show

# Multi-Layer Percepton MLP
# def mlp_clf():
#     print(colored("------MLP Classification-------", 'red'))

#     # Build classifier
#     clf_nn = MLPClassifier(
#         alpha=1e-5, hidden_layer_sizes=(1000, 5), max_iter=1000, random_state=1)

#     print("Training the MLP Classifier.......")

#     # start timer
#     starttime = timeit.default_timer()  # start timer

#     # train
#     clf_nn.fit(train_x, train_Y)

#     print("The time difference is :", timeit.default_timer() - starttime)

#     print("Predicting test data.......")

#     # predict
#     nn_pred = clf_nn.predict(test_x)

#     # results
#     c_matrix = confusion_matrix(test_Y, nn_pred)
#     error = zero_one_loss(test_Y, nn_pred)
#     score = accuracy_score(test_Y, nn_pred)

#     # display results
#     print('Confusion Matrix\n---------------------------\n', c_matrix)
#     print('---------------------------')
#     print("Error: {:.4f}%".format(error * 100))
#     print("Accuracy Score: {:.4f}%".format(score * 100))
#     print(classification_report(test_Y, nn_pred))
#     print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

#     # Plot non-normalized confusion matrix
#     disp = plot_confusion_matrix(clf_nn, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
#                                  xticks_rotation='horizontal')
#     plt.title("Confusion Matrix for Neural Network")

#     plt.show()


if __name__ == "__main__":

    print("------Logistic Regression Classification-------")

    # logistic regression classifier
    clf_lr = LogisticRegression(
        C=1e5, random_state=0
    )

    # start timer
    starttime = timeit.default_timer()

    print("Training the Logistic Regression Classifier.......")

    # train the model
    clf_lr = clf_lr.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    pred_y = clf_lr.predict(test_x)

    # get results
    c_matrix = confusion_matrix(test_Y, pred_y)
    error = zero_one_loss(test_Y, pred_y)
    score = accuracy_score(test_Y, pred_y)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, pred_y))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(clf_lr, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')

    plt.title("Confusion Matrix for Logistic Regression")
    utils.set_initial_params(clf_lr)

    class FlowerClient(fl.client.NumPyClient):

        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(clf_lr)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # logistic_reg_clf()
    # mlp_clf()
    fl.client.start_numpy_client("0.0.0.0:9000", client=FlowerClient())
