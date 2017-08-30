# coding: utf-8

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

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

sgd_clf = SGDClassifier(random_state=42, max_iter=200)

# sgd_clf.fit(X_train, y_train_5)

# # ===============sample test===============
# idx = 5500
# print('index', idx)
# print('predicted', sgd_clf.predict([X_test[ idx ]]))
# print('expected', y_test_5[idx])

# plot_digit(X_test[ idx ])
# plt.show()

# accuracy = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print('accuracy', accuracy)

# # ===============confusion matrix====================
# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# cm = confusion_matrix(y_train_5, y_train_pred)
# print('confusion matrix', cm)

# # ===============precision and recall===============
# y_predict = sgd_clf.predict(X_train)
# print('precision', precision_score(y_train_5, y_predict))
# print('recall', recall_score(y_train_5, y_predict))

# # ===============random forest prediction===============
# forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
#                                     method="predict_proba")
# print(y_probas_forest)


# ===============SGD for multiclass classification===============
# sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([X_train[3000]]), y_train[3000])



