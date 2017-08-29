
# coding: utf-8

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# to make this notebook's output stable across runs
np.random.seed(42)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()
# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.copy()
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)

# # # Select and train a model 
# from sklearn.linear_model import LinearRegression

# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)

# # let's try the full pipeline on a few training instances
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)

# print("Predictions:", lin_reg.predict(some_data_prepared))
# # Compare against the actual values:
# print("Labels:", list(some_labels))

# some_data_prepared

# from sklearn.metrics import mean_squared_error

# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# lin_rmse

# from sklearn.metrics import mean_absolute_error

# lin_mae = mean_absolute_error(housing_labels, housing_predictions)
# lin_mae

# from sklearn.tree import DecisionTreeRegressor

# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(housing_prepared, housing_labels)

# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# tree_rmse
# # # Fine-tune your model
# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)

# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())

# display_scores(tree_rmse_scores)

# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                              scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores)

# from sklearn.ensemble import RandomForestRegressor

# forest_reg = RandomForestRegressor(random_state=42)
# forest_reg.fit(housing_prepared, housing_labels)

# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# forest_rmse = np.sqrt(forest_mse)
# forest_rmse

# from sklearn.model_selection import cross_val_score

# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                                 scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

# scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# pd.Series(np.sqrt(-scores)).describe()

# from sklearn.svm import SVR

# svm_reg = SVR(kernel="linear")
# svm_reg.fit(housing_prepared, housing_labels)
# housing_predictions = svm_reg.predict(housing_prepared)
# svm_mse = mean_squared_error(housing_labels, housing_predictions)
# svm_rmse = np.sqrt(svm_mse)
# svm_rmse

# from sklearn.model_selection import GridSearchCV

# param_grid = [
#     # try 12 (3×4) combinations of hyperparameters
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     # then try 6 (2×3) combinations with bootstrap set as False
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
#   ]

# forest_reg = RandomForestRegressor(random_state=42)
# # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error')
# grid_search.fit(housing_prepared, housing_labels)
# # The best hyperparameter combination found:
# grid_search.best_params_

# grid_search.best_estimator_
# # Let's look at the score of each hyperparameter combination tested during the grid search:
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)

# pd.DataFrame(grid_search.cv_results_)

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint

# param_distribs = {
#         'n_estimators': randint(low=1, high=200),
#         'max_features': randint(low=1, high=8),
#     }

# forest_reg = RandomForestRegressor(random_state=42)
# rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                 n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
# rnd_search.fit(housing_prepared, housing_labels)

# cvres = rnd_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)

# feature_importances = grid_search.best_estimator_.feature_importances_
# feature_importances

# extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_one_hot_attribs = list(encoder.classes_)
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# sorted(zip(feature_importances, attributes), reverse=True)

# final_model = grid_search.best_estimator_

# X_test = strat_test_set.drop("median_house_value", axis=1)
# y_test = strat_test_set["median_house_value"].copy()

# X_test_prepared = full_pipeline.transform(X_test)
# final_predictions = final_model.predict(X_test_prepared)

# final_mse = mean_squared_error(y_test, final_predictions)
# final_rmse = np.sqrt(final_mse)

# final_rmse
# # # Extra material

# # ## Label Binarizer hack
# # `LabelBinarizer`'s `fit_transform()` method only accepts one parameter `y` (because it was meant for labels, not predictors), so it does not work in a pipeline where the final estimator is a supervised estimator because in this case its `fit()` method takes two parameters `X` and `y`.
# # 
# # This hack creates a supervision-friendly `LabelBinarizer`.
# class SupervisionFriendlyLabelBinarizer(LabelBinarizer):
#     def fit_transform(self, X, y=None):
#         return super(SupervisionFriendlyLabelBinarizer, self).fit_transform(X)

# # Replace the Labelbinarizer with a SupervisionFriendlyLabelBinarizer
# cat_pipeline.steps[1] = ("label_binarizer", SupervisionFriendlyLabelBinarizer())

# # Now you can create a full pipeline with a supervised predictor at the end.
# full_pipeline_with_predictor = Pipeline([
#         ("preparation", full_pipeline),
#         ("linear", LinearRegression())
#     ])

# full_pipeline_with_predictor.fit(housing, housing_labels)
# full_pipeline_with_predictor.predict(some_data)
# # ## Model persistence using joblib
# my_model = full_pipeline_with_predictor

# from sklearn.externals import joblib
# joblib.dump(my_model, "my_model.pkl") # DIFF
# #...
# my_model_loaded = joblib.load("my_model.pkl") # DIFF
# # ## Example SciPy distributions for `RandomizedSearchCV`
# from scipy.stats import geom, expon
# geom_distrib=geom(0.5).rvs(10000, random_state=42)
# expon_distrib=expon(scale=1).rvs(10000, random_state=42)
# plt.hist(geom_distrib, bins=50)
# plt.show()
# plt.hist(expon_distrib, bins=50)
# plt.show()
# # # Exercise solutions

# # ## 1.

# # Question: Try a Support Vector Machine regressor (`sklearn.svm.SVR`), with various hyperparameters such as `kernel="linear"` (with various values for the `C` hyperparameter) or `kernel="rbf"` (with various values for the `C` and `gamma` hyperparameters). Don't worry about what these hyperparameters mean for now. How does the best `SVR` predictor perform?
# from sklearn.model_selection import GridSearchCV

# param_grid = [
#         {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
#         {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
#          'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
#     ]

# svm_reg = SVR()
# grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
# grid_search.fit(housing_prepared, housing_labels)
# # The best model achieves the following score (evaluated using 5-fold cross validation):
# negative_mse = grid_search.best_score_
# rmse = np.sqrt(-negative_mse)
# rmse
# # That's much worse than the `RandomForestRegressor`. Let's check the best hyperparameters found:
# grid_search.best_params_
# # The linear kernel seems better than the RBF kernel. Notice that the value of `C` is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for `C` (removing the smallest values), because it is likely that higher values of `C` will be better.

# # ## 2.

# # Question: Try replacing `GridSearchCV` with `RandomizedSearchCV`.
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import expon, reciprocal

# # see https://docs.scipy.org/doc/scipy-0.19.0/reference/stats.html
# # for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# # Note: gamma is ignored when kernel is "linear"
# param_distribs = {
#         'kernel': ['linear', 'rbf'],
#         'C': reciprocal(20, 200000),
#         'gamma': expon(scale=1.0),
#     }

# svm_reg = SVR()
# rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
#                                 n_iter=50, cv=5, scoring='neg_mean_squared_error',
#                                 verbose=2, n_jobs=4, random_state=42)
# rnd_search.fit(housing_prepared, housing_labels)
# # The best model achieves the following score (evaluated using 5-fold cross validation):
# negative_mse = rnd_search.best_score_
# rmse = np.sqrt(-negative_mse)
# rmse
# # Now this is much closer to the performance of the `RandomForestRegressor` (but not quite there yet). Let's check the best hyperparameters found:
# rnd_search.best_params_
# # This time the search found a good set of hyperparameters for the RBF kernel. Randomized search tends to find better hyperparameters than grid search in the same amount of time.

# # Let's look at the exponential distribution we used, with `scale=1.0`. Note that some samples are much larger or smaller than 1.0, but when you look at the log of the distribution, you can see that most values are actually concentrated roughly in the range of exp(-2) to exp(+2), which is about 0.1 to 7.4.
# expon_distrib = expon(scale=1.)
# samples = expon_distrib.rvs(10000, random_state=42)
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.title("Exponential distribution (scale=1.0)")
# plt.hist(samples, bins=50)
# plt.subplot(122)
# plt.title("Log of this distribution")
# plt.hist(np.log(samples), bins=50)
# plt.show()
# # The distribution we used for `C` looks quite different: the scale of the samples is picked from a uniform distribution within a given range, which is why the right graph, which represents the log of the samples, looks roughly constant. This distribution is useful when you don't have a clue of what the target scale is:
# reciprocal_distrib = reciprocal(20, 200000)
# samples = reciprocal_distrib.rvs(10000, random_state=42)
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.title("Reciprocal distribution (scale=1.0)")
# plt.hist(samples, bins=50)
# plt.subplot(122)
# plt.title("Log of this distribution")
# plt.hist(np.log(samples), bins=50)
# plt.show()
# # The reciprocal distribution is useful when you have no idea what the scale of the hyperparameter should be (indeed, as you can see on the figure on the right, all scales are equally likely, within the given range), whereas the exponential distribution is best when you know (more or less) what the scale of the hyperparameter should be.

# # ## 3.

# # Question: Try adding a transformer in the preparation pipeline to select only the most important attributes.
# from sklearn.base import BaseEstimator, TransformerMixin

# def indices_of_top_k(arr, k):
#     return np.sort(np.argpartition(np.array(arr), -k)[-k:])

# class TopFeatureSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, feature_importances, k):
#         self.feature_importances = feature_importances
#         self.k = k
#     def fit(self, X, y=None):
#         self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
#         return self
#     def transform(self, X):
#         return X[:, self.feature_indices_]
# # Note: this feature selector assumes that you have already computed the feature importances somehow (for example using a `RandomForestRegressor`). You may be tempted to compute them directly in the `TopFeatureSelector`'s `fit()` method, however this would likely slow down grid/randomized search since the feature importances would have to be computed for every hyperparameter combination (unless you implement some sort of cache).

# # Let's define the number of top features we want to keep:
# k = 5
# # Now let's look for the indices of the top k features:
# top_k_feature_indices = indices_of_top_k(feature_importances, k)
# top_k_feature_indices

# np.array(attributes)[top_k_feature_indices]
# # Let's double check that these are indeed the top k features:
# sorted(zip(feature_importances, attributes), reverse=True)[:k]
# # Looking good... Now let's create a new pipeline that runs the previously defined preparation pipeline, and adds top k feature selection:
# preparation_and_feature_selection_pipeline = Pipeline([
#     ('preparation', full_pipeline),
#     ('feature_selection', TopFeatureSelector(feature_importances, k))
# ])

# housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
# # Let's look at the features of the first 3 instances:
# housing_prepared_top_k_features[0:3]
# # Now let's double check that these are indeed the top k features:
# housing_prepared[0:3, top_k_feature_indices]
# # Works great!  :)

# # ## 4.

# # Question: Try creating a single pipeline that does the full data preparation plus the final prediction.
# prepare_select_and_predict_pipeline = Pipeline([
#     ('preparation', full_pipeline),
#     ('feature_selection', TopFeatureSelector(feature_importances, k)),
#     ('svm_reg', SVR(**rnd_search.best_params_))
# ])

# prepare_select_and_predict_pipeline.fit(housing, housing_labels)
# # Let's try the full pipeline on a few instances:
# some_data = housing.iloc[:4]
# some_labels = housing_labels.iloc[:4]

# print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
# print("Labels:\t\t", list(some_labels))
# # Well, the full pipeline seems to work fine. Of course, the predictions are not fantastic: they would be better if we used the best `RandomForestRegressor` that we found earlier, rather than the best `SVR`.

# # ## 5.

# # Question: Automatically explore some preparation options using `GridSearchCV`.
# param_grid = [
#         {'preparation__num_pipeline__imputer__strategy': ['mean', 'median', 'most_frequent'],
#          'feature_selection__k': [3, 4, 5, 6, 7]}
# ]

# grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
#                                 scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
# grid_search_prep.fit(housing, housing_labels)

# grid_search_prep.best_params_
# # Great! It seems that we had the right imputer stragegy (mean), and apparently only the top 7 features are useful (out of 9), the last 2 seem to just add some noise.
# housing.shape
# # Congratulations! You already know quite a lot about Machine Learning. :)
