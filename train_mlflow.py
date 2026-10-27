import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Create or select an experiment
mlflow.set_experiment("Iris Classification")

# 2. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Define model parameters
n_estimators = 100
max_depth = 5

# 5. Start an MLflow run
with mlflow.start_run():

    # Record the parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Record the metric
    mlflow.log_metric("accuracy", accuracy)

    print("Model training completed!")
    print("Accuracy:", accuracy)
    print("MLflow run ID:", mlflow.active_run().info.run_id)

print("Experiment tracking completed!")
