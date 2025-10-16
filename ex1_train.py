import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# Set up MLflow experiment
mlflow.set_experiment("iris_classification")

# LOad iris dataset and split into train/test sets
iris= load_iris()
# print(iris) #loaded dataset
# print(iris.data)     #features
# print(iris.target)   #target labels

X_train,X_test, y_train, y_test= train_test_split(
    iris.data,   #features
    iris.target,   #target
    test_size=0.2,   #percentage of data to be tested
    random_state=78   # for reproducibility
)
# print("X_train:", X_train)
# print("y_train:", y_train)
# print("X_test:", X_test)
# print("y_test", y_test)

# STart ML flow
with mlflow.start_run():
    # Train model
    model=LogisticRegression(max_iter=200)
    # Fit the model
    model.fit(X_train,y_train)

    # Make predictions and calculate accuracy
    preds= model.predict(X_test) #remaining 30 samples
    # Preds are the targeted targets (0/1/2) for test set
    acc= accuracy_score(y_test,preds)

    # LOg to MLflow
    mlflow.log_param("model_type","logistic_regresssion")
    mlflow.log_metric("accuracy",acc)
    mlflow.sklearn.log_model(model,"model")

    print(f"Run logged to MLflow Accuracy: {acc:.3f}")
 
