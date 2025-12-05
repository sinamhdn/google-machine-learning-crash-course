import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_dataset = pd.read_csv("./datasets/california_housing_train.csv")

print(f"\nDATASET DESCRIPTION:\n{training_dataset.describe()}\n")

print(
    """The following columns might contain outliers:

  * total_rooms
  * total_bedrooms
  * population
  * households
  * possibly, median_income

In all of those columns:

  * the standard deviation is almost as high as the mean
  * the delta between 75% and max is much higher than the
      delta between min and 25%."""
)

# print(f"{outlier for value in training_dataset.columns.values if value.max-value.}")
