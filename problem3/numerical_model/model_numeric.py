import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns



def plot_roc_curve(model, X_test, y_test, model_name):
    # Checking if model has predict_proba method (for RandomForest, Neural Networks)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1] 
    else:
        if isinstance(model, VotingClassifier):
            probas = [clf.predict_proba(X_test)[:, 1] for clf in model.estimators_]
            y_prob = np.mean(probas, axis=0)  
        else:
            y_prob = model.decision_function(X_test)  
    
    # ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random guess)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()



# Training and test data
training_data = pd.read_csv('./data/pulsar_test.csv')
testing_data = pd.read_csv('./data/pulsar_training.csv')


# Cleaning Data
training_data.drop(columns=['Unnamed: 0'], inplace=True)
testing_data.drop(columns=['Unnamed: 0'], inplace=True)
training_data.dropna(inplace=True)
testing_data.dropna(inplace=True)


# Splitting into Features and Target
X_train = training_data.iloc[:, :-1]
y_train = training_data.iloc[:, -1]
X_test = testing_data.iloc[:, :-1]
y_test = testing_data.iloc[:, -1]


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 1. SVM Model with Hyperparameter Tuning
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [3, 4, 5],
    'coef0': [0, 0.1, 1, 10],
    'shrinking': [True, False],
    'tol': [1e-3, 1e-4, 1e-5],
    'max_iter': [-1, 2000, 2500],
    'cache_size': [200, 500, 1000]
}

svm_model = SVC(probability=True)
svm_search = RandomizedSearchCV(svm_model, svm_param_grid, n_iter=10, cv=2, n_jobs=-1, verbose=2, random_state=42)
svm_search.fit(X_train_scaled, y_train)
best_svm_model = svm_search.best_estimator_



# 2. Random Forest Model with Hyperparameter Tuning
rf_param_grid = {
    'n_estimators': np.arange(100, 501, 100),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, n_iter=15, cv=2, n_jobs=-1, verbose=2, random_state=42)
rf_model.fit(X_train_scaled, y_train)
best_rf_model = rf_model.best_estimator_



# 3. Neural Network Model with Hyperparameter Tuning
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'tol': [1e-4, 1e-5],
    'early_stopping': [True],
    'validation_fraction': [0.1, 0.2],
    'n_iter_no_change': [10, 20],
    'max_iter': [500, 1000, 3000]
}

mlp_model = RandomizedSearchCV(MLPClassifier(random_state=42), mlp_param_grid, n_iter=15, cv=2, n_jobs=-1, verbose=2, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
best_mlp_model = mlp_model.best_estimator_



# 4. Voting Classifier( SVM + RF + MLP )
voting_clf = VotingClassifier(
    estimators=[('svm', best_svm_model), ('rf', best_rf_model), ('mlp', best_mlp_model)], 
    voting='hard', 
    weights=[8, 2, 3]
)
voting_clf.fit(X_train_scaled, y_train)


# Function to evaluate models and display results
def evaluate_model(model_name, y_true, y_pred):
    print(f"{model_name} Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"{model_name} Classification Report:\n{classification_report(y_true, y_pred)}")
    print(f"{model_name} Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap='Blues', xticklabels=['Non-pulsar', 'Pulsar'], yticklabels=['Non-pulsar', 'Pulsar'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# ROC Curve
plot_roc_curve(best_svm_model, X_test_scaled, y_test, "SVM")
plot_roc_curve(best_rf_model, X_test_scaled, y_test, "Random Forest")
plot_roc_curve(best_mlp_model, X_test_scaled, y_test, "Neural Network")
plot_roc_curve(voting_clf, X_test_scaled, y_test, "Voting Classifier")


# Prediction by each Model
y_pred_svm = best_svm_model.predict(X_test_scaled)
y_pred_rf = best_rf_model.predict(X_test_scaled)
y_pred_mlp = best_mlp_model.predict(X_test_scaled)
y_pred_voting = voting_clf.predict(X_test_scaled)


# Evaluating models
evaluate_model('SVM Model', y_test, y_pred_svm)
evaluate_model('Random Forest Model', y_test, y_pred_rf)
evaluate_model('Neural Network Model', y_test, y_pred_mlp)
evaluate_model('Voting Classifier', y_test, y_pred_voting)
