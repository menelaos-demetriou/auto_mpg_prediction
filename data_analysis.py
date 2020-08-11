import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dython.nominal import associations
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

    # Drop car_name feature since it has too many unique values
    # TODO: Create etl to map car names to model
    data = data.drop(["car_name"], axis=1)

    # Need to look at the target variable
    print(data.mpg.describe())

    # Print Distribution of the target variable
    sns.distplot(data.mpg)
    plt.savefig("plots/mpg_distribution.jpg")
    plt.show()

    # Plot all numerical features against each other
    sns.pairplot(data, vars=["displacement", "horsepower", "weight", "acceleration"], hue="cylinders")
    plt.savefig("plots/pair_plot_cylinders.jpg")
    sns.pairplot(data, vars=["displacement", "horsepower", "weight", "acceleration"], hue="origin")
    plt.savefig("plots/pair_plot_origin.jpg")
    sns.pairplot(data, vars=["displacement", "horsepower", "weight", "acceleration"], hue="model_year")
    plt.savefig("plots/pair_plot_model_year.jpg")
    plt.show()

    # Plot categorical values against the target variable
    sns.boxplot(x="cylinders", y="mpg", data=data)
    plt.axhline(data.mpg.mean(), color='r', linestyle='dashed', linewidth=2)
    plt.savefig("plots/box_plot_cylinders.jpg")
    plt.show()
    sns.boxplot(x="origin", y="mpg", data=data)
    plt.axhline(data.mpg.mean(), color='r', linestyle='dashed', linewidth=2)
    plt.savefig("plots/box_plot_origin.jpg")
    plt.show()
    sns.boxplot(x="model_year", y="mpg", data=data)
    plt.axhline(data.mpg.mean(), color='r', linestyle='dashed', linewidth=2)
    plt.savefig("plots/box_plot_model_year.jpg")
    plt.show()

    # Get correlation plot
    associations(data, figsize=(15, 15), cmap="viridis")
    plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
