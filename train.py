import pickle
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
plt.style.use('ggplot')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """ Generate a simple plot of the test and traning learning curve. 
    Parameters ---------- 
    estimator : object type that implements the "fit" and "predict" methods An object of that type
    which is cloned for each validation.
    title : string Title for the chart. 
    X : array-like, shape (n_samples, n_features) Training vector,
    where n_samples is the number of samples and n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features),
    optional Target relative to X for classification or regression; None for unsupervised learning.
    ylim : tuple, shape (y_min, y_max), optional Defines minimum and maximum y values plotted.
    cv : integer, cross-validation generator,
    n_jobs : integer, optional Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            scoring="neg_root_mean_squared_error",
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    return plt


def main():
    data = pd.read_csv("data/auto-mpg.data-original", sep="\s+", header=None,
                       names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                              "acceleration", "model_year", "origin", "car_name"])
    # Drop na columns
    data = data.dropna()

    # Drop car_name feature since it has too many unique values
    # TODO: Create etl to map car names to model
    data = data.drop(["car_name"], axis=1)

    # Create power-to-weigth attribute
    data['power_to_weight'] = ((data.horsepower * 0.7457) / data.weight)

    num_attribs = ["displacement", "horsepower", "weight", "acceleration", "power_to_weight"]
    cat_attribs = ["cylinders", "origin", "model_year"]

    # Create preprocess transformations
    preprocess = ColumnTransformer([("num", StandardScaler(), num_attribs),
                                   ("cat", OneHotEncoder(sparse=False), cat_attribs),
                                   ])

    # Shuffle the dataset
    data = data.sample(frac=1)
    X = data.loc[:, ~data.columns.isin(['mpg'])].copy()
    y = data["mpg"].copy()

    X_transformed = preprocess.fit_transform(X)

    # Split data to train validation and test set
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.15, random_state=0)

    search_space = [
        {"regressor": [SGDRegressor()],
         "regressor__penalty": ['l2'],
         "regressor__loss": ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
         },
        {"regressor": [GradientBoostingRegressor()],
         "regressor__n_estimators": [100],
         "regressor__learning_rate": [0.1, 0.05, 0.02],
         "regressor__max_depth":[4, 6],
         "regressor__min_samples_leaf":[3, 5, 9, 17],
         "regressor__max_features":[1.0, 0.3]
        },
        {"regressor": [RandomForestRegressor()],
         "regressor__n_estimators": [10, 100, 1000],
         "regressor__max_depth": [5, 8, 15, 25, 30, None],
         "regressor__min_samples_leaf": [1, 2, 5, 10, 15, 100],
         "regressor__max_leaf_nodes": [2, 5, 10]
         }
        ,
        {"regressor": [SVR()],
         "regressor__kernel": ['rbf'],
         "regressor__gamma": [1e-3, 1e-4],
         "regressor__C": [1, 10, 100, 1000]
         },
        {"regressor": [SVR()],
         "regressor__kernel": ['linear'],
         "regressor__C": [1, 10, 100, 1000]
         }
    ]

    estimator = Pipeline([("regressor", SGDRegressor())])

    # Performing grid search on all classifiers and hyperparameter combinations
    gridsearch = GridSearchCV(estimator, search_space, cv=5, verbose=1, scoring="neg_root_mean_squared_error",
                              n_jobs=-1).fit(X_train, y_train)

    print("The best score is : %.2f" % gridsearch.best_score_)
    print("Best estimator: ", gridsearch.best_params_)

    plot_learning_curve(gridsearch.best_estimator_, "Learning curve", X_train, y_train, cv=5, n_jobs=-1)
    plt.savefig("plots/learning_curve.jpg")
    plt.show()

    print("Saving best estimator for evaluations on test dataset")
    filename = "model/best_estimator.sav"
    pickle.dump(gridsearch.best_estimator_, open(filename, 'wb'))

    print("Load model and perform evaluations")
    best_model = pickle.load(open(filename, "rb"))

    y_test_pred = best_model.predict(X_test)

    print("Mean squared error is %.2f" % sqrt(mean_squared_error(y_test, y_test_pred)))


if __name__ == "__main__":
    main()
