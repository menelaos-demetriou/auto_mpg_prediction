import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def main():
    # Read csv file and prepare dataframe
    data = pd.read_csv("data/auto-mpg.data-original", sep="\s+", header=None,
                       names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                              "acceleration", "model_year", "origin", "car_name"])

    print(data.info())
    print(data.shape)

    # Find How many null entries
    print(data.isnull().sum())
    # Drop na columns
    data = data.dropna()

    # Need to look at the target variable
    print(data.mpg.describe())

    sns.distplot(data['mpg'])
    plt.show()


    print("Hello")


if __name__ == "__main__":
    # execute only if run as a script
    main()