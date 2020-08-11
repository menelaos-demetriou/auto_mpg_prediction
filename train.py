import pickle
import pandas as pd
from math import sqrt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


def feature_selection(data):
    y = data["mpg"].copy()
    X = data.loc[:, ~data.columns.isin(["mpg"])].copy()

    pca = PCA(n_components=3)
    x_reduced = pca.fit_transform(X)
    return pd.concat([pd.DataFrame(x_reduced), pd.Series(y, name="target")], axis=1)


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

    data_reduced = feature_selection(data)

    # Split data to train validation and test set
    X_train, X_test, y_train, y_test = train_test_split(data_reduced.loc[:, ~data_reduced.columns.isin(['mpg'])].copy(),
                                                        data_reduced["mpg"].copy(), test_size=0.15,
                                                        random_state=18)

    num_attribs = ["displacement", "horsepower", "weight", "acceleration", "power_to_weight"]
    cat_attribs = ["cylinders", "origin", "model_year"]
    # Create preprocess transformations
    preprocess = ColumnTransformer([("num", StandardScaler(), num_attribs),
                                   ("cat", OneHotEncoder(sparse=False), cat_attribs),
                                   ])




if __name__ == "__main__":
    main()
