import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yaml

# load the reference data
dataset_folder = "datasets/financial"
X_train_reference = pd.read_csv(f"{dataset_folder}/X_train_reference.csv")
X_test_reference = pd.read_csv(f"{dataset_folder}/X_test_reference.csv")
y_train_reference = pd.read_csv(f"{dataset_folder}/y_train_reference.csv").squeeze("columns")
y_test_reference = pd.read_csv(f"{dataset_folder}/y_test_reference.csv").squeeze("columns")

# load the new data
X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")

# Train and evaluate reference model
model_reference = RandomForestClassifier(random_state=42)
model_reference.fit(X_train_reference, y_train_reference)

# Test reference model on reference test set
y_pred_reference = model_reference.predict(X_test_reference)
ref_score_reference = accuracy_score(y_test_reference, y_pred_reference)
print(f'Model trained and evaluated on the reference distribution: {ref_score_reference}')
metrics = {'model_reference': {'score_reference_data': float(ref_score_reference)}}

# Test reference model on new test set
y_pred_new = model_reference.predict(X_test_new)
ref_score_new = accuracy_score(y_test_new, y_pred_new)
print(f'Reference model evaluated on the new distribution: {ref_score_new}')
metrics['model_reference']['score_new_data'] = float(ref_score_new)

# Calculate average score for reference model
ref_score_average = (ref_score_reference + ref_score_new) / 2
print(f'Average score of reference model: {ref_score_average}')
metrics['model_reference']['score_average'] = float(ref_score_average)

print("\nTraining new model on combined data...")

# Combine datasets
X_train = pd.concat([X_train_reference, X_train_new])
y_train = pd.concat([y_train_reference, y_train_new])

# Train new model on combined dataset
model_new = RandomForestClassifier(random_state=42)
model_new.fit(X_train, y_train)

# Test new model on reference test set
y_pred_reference_new = model_new.predict(X_test_reference)
new_score_reference = accuracy_score(y_test_reference, y_pred_reference_new)
print(f'New model evaluated on reference distribution: {new_score_reference}')
metrics['model_new'] = {'score_reference_data': float(new_score_reference)}

# Test new model on new test set
y_pred_new_new = model_new.predict(X_test_new)
new_score_new = accuracy_score(y_test_new, y_pred_new_new)
print(f'New model evaluated on new distribution: {new_score_new}')
metrics['model_new']['score_new_data'] = float(new_score_new)

# Calculate average score for new model
new_score_average = (new_score_reference + new_score_new) / 2
print(f'Average score of new model: {new_score_average}')
metrics['model_new']['score_average'] = float(new_score_average)

# Calculate score difference
score_difference = new_score_average - ref_score_average
print(f'\nScore difference: {score_difference}')
metrics['difference_score_averages'] = {'score_average': float(score_difference)}

# Save metrics to yaml file
with open('fast_graph_metrics.yaml', 'w') as f:
    yaml.dump(metrics, f)