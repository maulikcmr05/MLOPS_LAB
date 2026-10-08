import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# 1. MLflow Tracking Configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("MLflow Tracking URI:", mlflow.get_tracking_uri())


# 2. Load Iris Dataset
data = load_iris()

X = data.data
y = data.target

print("\nDataset loaded successfully")
print("Total samples:", len(X))
print("Number of features:", X.shape[1])
print("Number of classes:", len(data.target_names))


# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Create / Select MLflow Experiment
experiment_name = "Iris_Classification_Tuning"

mlflow.set_experiment(experiment_name)

print("\nMLflow Experiment:", experiment_name)


# 5. Define Hyperparameter Grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 3, 4]
}


# 6. Run Experiments
for params in ParameterGrid(param_grid):

    print("\n" + "=" * 60)
    print("Starting experiment")
    print("Parameters:", params)
    print("=" * 60)

    # Start MLflow Run
    with mlflow.start_run():

        # Log Parameters
        mlflow.log_param(
            "n_estimators",
            params["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            params["max_depth"]
        )

        # Create Random Forest Model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        # Train Model
        model.fit(X_train, y_train)

        # Prediction
        predictions = model.predict(X_test)

        # Calculate Metrics
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro"
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro"
        )

        # Log Metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "f1_macro",
            f1
        )

        mlflow.log_metric(
            "precision_macro",
            precision
        )

        mlflow.log_metric(
            "recall_macro",
            recall
        )

        # Log Model
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # Display Results
        print("\nResults:")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(
            "Run ID    :",
            mlflow.active_run().info.run_id
        )


print("\nAll experiments completed successfully.")