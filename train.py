import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


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
    data['Power_to_weight'] = ((data.horsepower * 0.7457) / data.weight)


if __name__ == "__main__":
    main()
