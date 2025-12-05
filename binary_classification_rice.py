import importlib
import subprocess
import sys
from typing import List, Tuple


def ensure_libraries_installed(libraries: List[Tuple[str, str]]):
    for package, name in libraries:
        try:
            importlib.import_module(name)
            f"✓ {name} is already installed"
        except:
            print(f"⏳ Installing {package}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                importlib.import_module(name)
                print(f"✓ Successfully installed and imported {name}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                sys.exit(1)
            except ImportError:
                print(f"❌ Failed to import {name} after installation")
                sys.exit(1)


required_libraries = [
    ("google-ml-edu==0.1.3", "ml_edu"),
    ("tensorflow~=2.18.0", "tensorflow"),
    ("keras~=3.8.0", "keras"),
    ("matplotlib~=3.10.0", "matplotlib"),
    ("plotly", "plotly"),
    ("numpy~=2.0.0", "numpy"),
    ("pandas~=2.2.0", "pandas"),
]
ensure_libraries_installed(required_libraries)

import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Ran the import statement.")

rice_dataset_raw = pd.read_csv("datasets/Rice_Cammeo_Osmancik.csv")

# @title
# Read and provide statistics on the dataset.
rice_dataset = rice_dataset_raw[
    [
        "Area",
        "Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Eccentricity",
        "Convex_Area",
        "Extent",
        "Class",
    ]
]

print("{}".format(rice_dataset.describe()))

print(
    "Min length of rice grains: {:.1f}".format(rice_dataset["Major_Axis_Length"].min())
)
print(
    "Max length of rice grains: {:.1f}".format(rice_dataset["Major_Axis_Length"].max())
)
print(
    "Standard deviation of largest rice grains perimeter from the mean: {:.1f}".format(
        (rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())
        / rice_dataset.Perimeter.std()
    )
)

for x_axis_data, y_axis_data in [
    ("Area", "Eccentricity"),
    ("Convex_Area", "Perimeter"),
    ("Major_Axis_Length", "Minor_Axis_Length"),
    ("Perimeter", "Extent"),
    ("Eccentricity", "Major_Axis_Length"),
]:
    px.scatter(rice_dataset, x=x_axis_data, y=y_axis_data, color="Class").show()

# @title Plot three features in 3D by entering their names and running this cell

x_axis_data = "Area"
y_axis_data = "Perimeter"
z_axis_data = "Class"
px.scatter_3d(
    rice_dataset,
    x=x_axis_data,
    y=y_axis_data,
    z=z_axis_data,
    color="Class",
).show()

# Calculate the Z-scores of each numerical column in the raw data and write
# them into a new DataFrame named df_norm.

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes("number").columns
normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std
normalized_dataset["Class"] = rice_dataset["Class"]
print("{}".format(normalized_dataset.head()))

keras.utils.set_random_seed(42)

normalized_dataset["Class_Bool"] = (normalized_dataset["Class"] == "Cammeo").astype(int)
normalized_dataset.sample(10)

number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)
shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]
test_data.head()

# to prevent label leakage
label_columns = ["Class", "Class_Bool"]
train_features = train_data.drop(columns=label_columns)
train_labels = train_data["Class_Bool"].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data["Class_Bool"].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data["Class_Bool"].to_numpy()

input_features = [
    "Eccentricity",
    "Major_Axis_Length",
    "Area",
]


# @title Define the functions that create and train a model.
def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """Create and compile a simple classification model."""
    model_inputs = [
        keras.Input(name=feature, shape=(1,)) for feature in settings.input_features
    ]
    # Use a Concatenate layer to assemble the different inputs into a single
    # tensor which will be given as input to the Dense layer.
    # For example: [input_1[0][0], input_2[0][0]]
    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    model_output = keras.layers.Dense(
        units=1, name="dense_layer", activation=keras.activations.sigmoid
    )(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)
    # Call the compile method to transform the layers into a model that
    # Keras can execute.  Notice that we're using a different loss
    # function for classification than for regression.
    model.compile(
        optimizer=keras.optimizers.RMSprop(settings.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    """Feed a dataset into the model in order to train it."""
    # The x parameter of keras.Model.fit can be a list of arrays, where
    # each array contains the data for one feature.
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }
    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )
    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


print("Defined the create_model and train_model functions.")

# Let's define our first experiment settings.
settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name="accuracy", threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name="precision", thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(name="recall", thresholds=settings.classification_threshold),
    keras.metrics.AUC(num_thresholds=100, name="auc"),
]

# Establish the model's topography.
model = create_model(settings, metrics)

# Train the model on the training set.
experiment = train_model("baseline", model, train_features, train_labels, settings)

# Plot metrics vs. epochs
ml_edu.results.plot_experiment_metrics(experiment, ["accuracy", "precision", "recall"])
ml_edu.results.plot_experiment_metrics(experiment, ["auc"])


def compare_train_validation(
    experiment: ml_edu.experiment.Experiment, validation_metrics: dict[str, float]
):
    print("Comparing metrics between train and validation:")
    for metric, validation_value in validation_metrics.items():
        print("------")
        print(f"Train {metric}: {experiment.get_final_metric_value(metric):.4f}")
        print(f"Validation {metric}:  {validation_value:.4f}")


# Evaluate validation metrics
validation_metrics = experiment.evaluate(validation_features, validation_labels)
compare_train_validation(experiment, validation_metrics)


all_input_features = [
    "Eccentricity",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Area",
    "Convex_Area",
    "Perimeter",
    "Extent",
]

settings_all_features = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.5,
    input_features=all_input_features,
)

# Modify the following definition of METRICS to generate
# not only accuracy and precision, but also recall:
metrics = [
    keras.metrics.BinaryAccuracy(
        name="accuracy",
        threshold=settings_all_features.classification_threshold,
    ),
    keras.metrics.Precision(
        name="precision",
        thresholds=settings_all_features.classification_threshold,
    ),
    keras.metrics.Recall(
        name="recall", thresholds=settings_all_features.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name="auc"),
]

# Establish the model's topography.
model_all_features = create_model(settings_all_features, metrics)

# Train the model on the training set.
experiment_all_features = train_model(
    "all features",
    model_all_features,
    train_features,
    train_labels,
    settings_all_features,
)

# Plot metrics vs. epochs
ml_edu.results.plot_experiment_metrics(
    experiment_all_features, ["accuracy", "precision", "recall"]
)
ml_edu.results.plot_experiment_metrics(experiment_all_features, ["auc"])

validation_metrics_all_features = experiment_all_features.evaluate(
    validation_features,
    validation_labels,
)
compare_train_validation(experiment_all_features, validation_metrics_all_features)

ml_edu.results.compare_experiment(
    [experiment, experiment_all_features],
    ["accuracy", "auc"],
    validation_features,
    validation_labels,
)

test_metrics_all_features = experiment_all_features.evaluate(
    test_features,
    test_labels,
)
for metric, test_value in test_metrics_all_features.items():
    print(f"Test {metric}:  {test_value:.4f}")
