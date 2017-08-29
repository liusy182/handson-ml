# coding: utf-8

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

# to make this notebook's output stable across runs
np.random.seed(42)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
print('X.shape={} y.shape={}'.format(X.shape, y.shape))


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# # # Binary classifier, classify 5 v.s. not 5
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42, tol=0.01)
sgd_clf.fit(X_train, y_train_5)

idx = 5500
print('index', idx)
print('predicted', sgd_clf.predict([X_test[ idx ]]))
print('expected', y_test_5[idx])
# plot_digit(X_test[ idx ])
# plt.show()
accuracy = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(accuracy)

# from sklearn.model_selection import StratifiedKFold
# from sklearn.base import clone

# skfolds = StratifiedKFold(n_splits=3, random_state=42)

# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = (y_train_5[train_index])
#     X_test_fold = X_train[test_index]
#     y_test_fold = (y_train_5[test_index])

#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))

# from sklearn.base import BaseEstimator
# class Never5Classifier(BaseEstimator):
#     def fit(self, X, y=None):
#         pass
#     def predict(self, X):
#         return np.zeros((len(X), 1), dtype=bool)

# never_5_clf = Never5Classifier()
# cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# from sklearn.model_selection import cross_val_predict

# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_train_5, y_train_pred)

# y_train_perfect_predictions = y_train_5

# confusion_matrix(y_train_5, y_train_perfect_predictions)

# from sklearn.metrics import precision_score, recall_score

# precision_score(y_train_5, y_train_pred)

# 4344 / (4344 + 1307)

# recall_score(y_train_5, y_train_pred)

# 4344 / (4344 + 1077)

# from sklearn.metrics import f1_score
# f1_score(y_train_5, y_train_pred)

# 4344 / (4344 + (1077 + 1307)/2)

# y_scores = sgd_clf.decision_function([some_digit])
# y_scores

# threshold = 0
# y_some_digit_pred = (y_scores > threshold)

# y_some_digit_pred

# threshold = 200000
# y_some_digit_pred = (y_scores > threshold)
# y_some_digit_pred

# y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
#                              method="decision_function")

# from sklearn.metrics import precision_recall_curve

# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
#     plt.xlabel("Threshold", fontsize=16)
#     plt.legend(loc="upper left", fontsize=16)
#     plt.ylim([0, 1])

# plt.figure(figsize=(8, 4))
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.xlim([-700000, 700000])
# save_fig("precision_recall_vs_threshold_plot")
# plt.show()

# (y_train_pred == (y_scores > 0)).all()

# y_train_pred_90 = (y_scores > 70000)

# precision_score(y_train_5, y_train_pred_90)

# recall_score(y_train_5, y_train_pred_90)

# def plot_precision_vs_recall(precisions, recalls):
#     plt.plot(recalls, precisions, "b-", linewidth=2)
#     plt.xlabel("Recall", fontsize=16)
#     plt.ylabel("Precision", fontsize=16)
#     plt.axis([0, 1, 0, 1])

# plt.figure(figsize=(8, 6))
# plot_precision_vs_recall(precisions, recalls)
# save_fig("precision_vs_recall_plot")
# plt.show()
# # # ROC curves


# from sklearn.metrics import roc_curve

# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)

# plt.figure(figsize=(8, 6))
# plot_roc_curve(fpr, tpr)
# save_fig("roc_curve_plot")
# plt.show()

# from sklearn.metrics import roc_auc_score

# roc_auc_score(y_train_5, y_scores)

# from sklearn.ensemble import RandomForestClassifier
# forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
#                                     method="predict_proba")

# y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower right", fontsize=16)
# save_fig("roc_curve_comparison_plot")
# plt.show()

# roc_auc_score(y_train_5, y_scores_forest)

# y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
# precision_score(y_train_5, y_train_pred_forest)

# recall_score(y_train_5, y_train_pred_forest)
# # # Multiclass classification


# sgd_clf.fit(X_train, y_train)
# sgd_clf.predict([some_digit])

# some_digit_scores = sgd_clf.decision_function([some_digit])
# some_digit_scores

# np.argmax(some_digit_scores)

# sgd_clf.classes_

# sgd_clf.classes_[5]

# from sklearn.multiclass import OneVsOneClassifier
# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, y_train)
# ovo_clf.predict([some_digit])

# len(ovo_clf.estimators_)

# forest_clf.fit(X_train, y_train)
# forest_clf.predict([some_digit])

# forest_clf.predict_proba([some_digit])

# cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# conf_mx

# def plot_confusion_matrix(matrix):
#     """If you prefer color and a colorbar"""
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(matrix)
#     fig.colorbar(cax)

# plt.matshow(conf_mx, cmap=plt.cm.gray)
# save_fig("confusion_matrix_plot", tight_layout=False)
# plt.show()

# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums

# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# save_fig("confusion_matrix_errors_plot", tight_layout=False)
# plt.show()

# cl_a, cl_b = 3, 5
# X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

# plt.figure(figsize=(8,8))
# plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# save_fig("error_analysis_digits_plot")
# plt.show()
# # # Multilabel classification


# from sklearn.neighbors import KNeighborsClassifier

# y_train_large = (y_train >= 7)
# y_train_odd = (y_train % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]

# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_multilabel)

# knn_clf.predict([some_digit])

# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
# f1_score(y_multilabel, y_train_knn_pred, average="macro")
# # # Multioutput classification


# noise = np.random.randint(0, 100, (len(X_train), 784))
# X_train_mod = X_train + noise
# noise = np.random.randint(0, 100, (len(X_test), 784))
# X_test_mod = X_test + noise
# y_train_mod = X_train
# y_test_mod = X_test

# some_index = 5500
# plt.subplot(121); plot_digit(X_test_mod[some_index])
# plt.subplot(122); plot_digit(y_test_mod[some_index])
# save_fig("noisy_digit_example_plot")
# plt.show()

# knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[some_index]])
# plot_digit(clean_digit)
# save_fig("cleaned_digit_example_plot")
# # # Extra material

# # ## Dummy (ie. random) classifier


# from sklearn.dummy import DummyClassifier
# dmy_clf = DummyClassifier()
# y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")
# y_scores_dmy = y_probas_dmy[:, 1]

# fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
# plot_roc_curve(fprr, tprr)
# # ## KNN classifier


# from sklearn.neighbors import KNeighborsClassifier
# knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
# knn_clf.fit(X_train, y_train)

# y_knn_pred = knn_clf.predict(X_test)

# from sklearn.metrics import accuracy_score
# accuracy_score(y_test, y_knn_pred)

# from scipy.ndimage.interpolation import shift
# def shift_digit(digit_array, dx, dy, new=0):
#     return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)

# plot_digit(shift_digit(some_digit, 5, 1, new=100))

# X_train_expanded = [X_train]
# y_train_expanded = [y_train]
# for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
#     shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
#     X_train_expanded.append(shifted_images)
#     y_train_expanded.append(y_train)

# X_train_expanded = np.concatenate(X_train_expanded)
# y_train_expanded = np.concatenate(y_train_expanded)
# X_train_expanded.shape, y_train_expanded.shape

# knn_clf.fit(X_train_expanded, y_train_expanded)

# y_knn_expanded_pred = knn_clf.predict(X_test)

# accuracy_score(y_test, y_knn_expanded_pred)

# ambiguous_digit = X_test[2589]
# knn_clf.predict_proba([ambiguous_digit])

# plot_digit(ambiguous_digit)
# # # Exercise solutions

# # **Coming soon**


