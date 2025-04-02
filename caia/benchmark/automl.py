from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import yaml
import time
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class DeterministicAutoML:
    """
    A deterministic AutoML implementation for fair comparison with agent-based approaches.
    This implementation uses fixed random seeds and predetermined model selection to ensure
    reproducible results.
    """
    
    def __init__(self, max_iterations=3, random_state=42):
        """
        Initialize the AutoML system.
        
        Args:
            max_iterations: Maximum number of improvement iterations to run
            random_state: Random seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.start_time = None
        self.iteration_times = []
        self.best_model = None
        self.best_pipeline = None
        self.best_score = 0.0
        self.improvement_history = []
        self.categorical_features = []
        self.numerical_features = []
        
    def _get_model_candidates(self, iteration=1):
        """
        Returns a fixed list of model candidates for each iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            List of (name, model, params_grid) tuples
        """
        # Common parameters for all iterations
        base_random_forest = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'random_state': [self.random_state]
        }
        
        base_gradient_boosting = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 5],
            'random_state': [self.random_state]
        }
        
        # First iteration: Try basic models with simple parameters
        if iteration == 1:
            return [
                ('random_forest', RandomForestClassifier(), base_random_forest),
                ('gradient_boosting', GradientBoostingClassifier(), base_gradient_boosting),
                ('logistic_regression', LogisticRegression(), {
                    'C': [0.1, 1.0, 10.0],
                    'random_state': [self.random_state]
                })
            ]
        
        # Second iteration: Enhanced parameter grids and new models
        elif iteration == 2:
            enhanced_rf = base_random_forest.copy()
            enhanced_rf.update({
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            })
            
            enhanced_gb = base_gradient_boosting.copy()
            enhanced_gb.update({
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5]
            })
            
            return [
                ('random_forest', RandomForestClassifier(), enhanced_rf),
                ('gradient_boosting', GradientBoostingClassifier(), enhanced_gb),
                ('adaboost', AdaBoostClassifier(), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.5, 1.0],
                    'random_state': [self.random_state]
                })
            ]
        
        # Third iteration: Focus on ensemble methods and advanced tuning
        else:
            return [
                ('gradient_boosting', GradientBoostingClassifier(), {
                    'n_estimators': [200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'random_state': [self.random_state]
                }),
                ('svm', SVC(), {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'probability': [True],
                    'random_state': [self.random_state]
                })
            ]
    
    def _create_preprocessor(self, categorical_features, numerical_features):
        """
        Create a deterministic preprocessing pipeline.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            
        Returns:
            ColumnTransformer preprocessor
        """
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
        
    def fit(self, X_train_old, y_train_old, X_test_old, y_test_old, 
            X_train_new, y_train_new, X_test_new, y_test_new, 
            categorical_features=None, numerical_features=None, 
            dataset_description=None):
        """
        Run the AutoML process to find the best model.
        
        Args:
            X_train_old: Original training features
            y_train_old: Original training labels
            X_test_old: Original test features
            y_test_old: Original test labels
            X_train_new: New training features
            y_train_new: New training labels
            X_test_new: New test features
            y_test_new: New test labels
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            dataset_description: Optional dataset metadata
            
        Returns:
            Dictionary with results and metrics
        """
        self.start_time = time.time()
        initial_code = self._generate_base_code()
        
        # Determine feature types if not provided
        if numerical_features is None and categorical_features is None:
            if dataset_description and 'NUMERICAL_FEATURES' in dataset_description:
                numerical_features = dataset_description.get('NUMERICAL_FEATURES', [])
                categorical_features = dataset_description.get('CATEGORICAL_FEATURES', [])
            else:
                # Try to infer feature types
                numerical_features = X_train_old.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X_train_old.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        print(f"AutoML initialized with {len(numerical_features)} numerical features and {len(categorical_features)} categorical features")
        print(f"Max iterations: {self.max_iterations}")
        
        # Train and evaluate base model (RandomForest) on old data
        base_model = RandomForestClassifier(random_state=self.random_state)
        base_model.fit(X_train_old, y_train_old)
        base_score_old = base_model.score(X_test_old, y_test_old)
        base_score_new = base_model.score(X_test_new, y_test_new)
        
        print(f"Base model - Old distribution score: {base_score_old:.4f}")
        print(f"Base model - New distribution score: {base_score_new:.4f}")
        
        # Initial metrics
        model_old_score = {
            'on_old_data': base_score_old,
            'on_new_data': base_score_new
        }
        
        # Combine old and new data for training
        X_train_combined = pd.concat([X_train_old, X_train_new])
        y_train_combined = pd.concat([y_train_old, y_train_new])
        
        # Keep track of the best model and scores
        self.best_score = 0
        best_combined_score = 0
        improvement_path = []
        
        # Run iteration
        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.time()
            print(f"\n\n{'='*20} ITERATION {iteration}/{self.max_iterations} {'='*20}\n")
            
            # Create preprocessor
            preprocessor = self._create_preprocessor(categorical_features, numerical_features)
            
            # Get model candidates for this iteration
            model_candidates = self._get_model_candidates(iteration)
            
            iteration_best_model = None
            iteration_best_score = 0
            iteration_best_pipeline = None
            iteration_results = {}
            
            # Try each model candidate
            for model_name, model, param_grid in model_candidates:
                print(f"\nTrying {model_name}...")
                
                # Create pipeline with preprocessing and model
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                # Create grid search
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid={f'classifier__{param}': values for param, values in param_grid.items()},
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                try:
                    # Fit the grid search
                    grid_search.fit(X_train_combined, y_train_combined)
                    
                    # Get best model
                    best_estimator = grid_search.best_estimator_
                    
                    # Evaluate on both test sets
                    old_score = best_estimator.score(X_test_old, y_test_old)
                    new_score = best_estimator.score(X_test_new, y_test_new)
                    
                    # Calculate combined score (prioritizing new distribution slightly)
                    combined_score = (old_score + new_score * 1.2) / 2.2
                    
                    print(f"  {model_name} - Old distribution: {old_score:.4f}, New distribution: {new_score:.4f}")
                    print(f"  Best params: {grid_search.best_params_}")
                    
                    # Update iteration results
                    iteration_results[model_name] = {
                        'model': best_estimator,
                        'pipeline': best_estimator,
                        'old_score': old_score,
                        'new_score': new_score,
                        'combined_score': combined_score,
                        'best_params': grid_search.best_params_
                    }
                    
                    # Update iteration best
                    if combined_score > iteration_best_score:
                        iteration_best_score = combined_score
                        iteration_best_model = best_estimator
                        iteration_best_pipeline = best_estimator
                        
                except Exception as e:
                    print(f"  Error with {model_name}: {str(e)}")
            
            # If we found a model this iteration
            if iteration_best_model is not None:
                best_result = max(iteration_results.items(), key=lambda x: x[1]['combined_score'])
                best_model_name = best_result[0]
                best_model_info = best_result[1]
                
                # Extract scores
                old_score = best_model_info['old_score']
                new_score = best_model_info['new_score']
                combined_score = best_model_info['combined_score']
                
                # Check if this is better than our overall best
                if combined_score > best_combined_score:
                    self.best_model = iteration_best_model
                    self.best_pipeline = iteration_best_pipeline
                    self.best_score = combined_score
                    best_combined_score = combined_score
                
                # Generate improved code
                improved_code = self._generate_improved_code(
                    model_name=best_model_name,
                    params=best_model_info['best_params'],
                    numerical_features=numerical_features,
                    categorical_features=categorical_features
                )
                
                # Format parameter string for display
                param_str = ', '.join([f"{k.split('__')[-1]}={v}" for k, v in best_model_info['best_params'].items()])
                
                # Create improvement entry
                improvement_entry = {
                    "iteration": iteration,
                    "code": improved_code,
                    "metrics": {
                        "old_distribution": old_score,
                        "new_distribution": new_score
                    },
                    "changes": [
                        f"Used {best_model_name} model",
                        f"Applied parameters: {param_str}",
                        "Applied feature preprocessing"
                    ],
                    "reflection": f"AutoML iteration {iteration} selected {best_model_name} with parameters: {param_str}",
                    "execution_time": time.time() - iteration_start
                }
                
                improvement_path.append(improvement_entry)
                self.improvement_history.append(improvement_entry)
                
                # Update model score for metrics tracking
                model_new_score = {
                    'on_old_data': old_score,
                    'on_new_data': new_score
                }
                
                # Record iteration time
                self.iteration_times.append({
                    'iteration': iteration,
                    'time': time.time() - iteration_start
                })
                
                print(f"\nIteration {iteration} complete in {time.time() - iteration_start:.2f} seconds")
                print(f"Best model: {best_model_name}")
                print(f"Old distribution: {old_score:.4f}, New distribution: {new_score:.4f}")
            else:
                print(f"No successful models in iteration {iteration}")
        
        # Calculate total runtime
        total_runtime = time.time() - self.start_time
        
        # Generate final code using best model from all iterations
        if self.best_model:
            final_code = self._generate_improved_code(
                model_name="best_model",
                params={},  # Will be extracted from the pipeline
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                pipeline=self.best_pipeline
            )
        else:
            final_code = initial_code
            
        # Prepare final metrics
        if improvement_path:
            final_metrics = improvement_path[-1]["metrics"]
        else:
            final_metrics = {
                "old_distribution": base_score_old,
                "new_distribution": base_score_new
            }
            
        # Prepare result YAML
        result = {
            "agent_name": "deterministic_automl",
            "initial_code": initial_code,
            "initial_metrics": {
                "old_distribution": base_score_old,
                "new_distribution": base_score_new
            },
            "improvement_path": improvement_path,
            "final_code": final_code,
            "final_metrics": final_metrics,
            "runtime_statistics": {
                "total_time_seconds": total_runtime,
                "iterations": len(improvement_path),
                "tokens_used": 0,  # Not applicable for AutoML
                "iteration_times": self.iteration_times,
                "evaluation_timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        return result
    
    def _generate_base_code(self):
        """Generate baseline code for the initial model"""
        return """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
dataset_folder = "datasets/financial"
X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")

# Train basic model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_old, y_train_old)

# Evaluate
test_score = accuracy_score(y_test_old, model.predict(X_test_old))
print(f"Test accuracy: {test_score:.4f}")
"""
    
    def _generate_improved_code(self, model_name, params, numerical_features, categorical_features, pipeline=None):
        """Generate improved code based on the best model found"""
        model_class = model_name.replace('_', ' ').title().replace(' ', '')
        
        # If we have a pipeline, extract actual parameter values
        if pipeline is not None:
            try:
                # Extract classifier parameters
                classifier = pipeline.named_steps['classifier']
                params = {param: getattr(classifier, param) for param in classifier.get_params()}
                # Clean up params that don't belong directly in the constructor
                params = {k: v for k, v in params.items() if not k.startswith('_') and not callable(v)}
            except Exception as e:
                print(f"Error extracting parameters from pipeline: {str(e)}")
        
        # Format parameters for code
        param_str = ",\n        ".join([f"{k}={repr(v)}" for k, v in params.items() if not k.startswith('classifier__')])
        
        # Determine imports based on model_name
        if model_name == 'random_forest':
            imports = "from sklearn.ensemble import RandomForestClassifier"
            model_init = f"RandomForestClassifier(\n        {param_str}\n    )"
        elif model_name == 'gradient_boosting':
            imports = "from sklearn.ensemble import GradientBoostingClassifier"
            model_init = f"GradientBoostingClassifier(\n        {param_str}\n    )"
        elif model_name == 'adaboost':
            imports = "from sklearn.ensemble import AdaBoostClassifier"
            model_init = f"AdaBoostClassifier(\n        {param_str}\n    )"
        elif model_name == 'logistic_regression':
            imports = "from sklearn.linear_model import LogisticRegression"
            model_init = f"LogisticRegression(\n        {param_str}\n    )"
        elif model_name == 'svm':
            imports = "from sklearn.svm import SVC"
            model_init = f"SVC(\n        {param_str}\n    )"
        else:
            # Generic case
            imports = f"from sklearn.ensemble import {model_class}"
            model_init = f"{model_class}(\n        {param_str}\n    )"
            
        # Add preprocessing imports if needed
        if numerical_features or categorical_features:
            imports += """
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline"""
            
        if numerical_features:
            imports += "\nfrom sklearn.preprocessing import StandardScaler"
            
        if categorical_features:
            imports += "\nfrom sklearn.preprocessing import OneHotEncoder"
        
        # Generate code
        code = f"""import pandas as pd
import yaml
{imports}
from sklearn.metrics import accuracy_score

# Initialize metrics dictionary
model_new_score = {{
    'on_new_data': 0.0,
    'on_old_data': 0.0
}}

# Load data
dataset_folder = "datasets/financial"
X_train_old = pd.read_csv(f"{{dataset_folder}}/X_train_old.csv")
X_test_old = pd.read_csv(f"{{dataset_folder}}/X_test_old.csv")
y_train_old = pd.read_csv(f"{{dataset_folder}}/y_train_old.csv").squeeze("columns")
y_test_old = pd.read_csv(f"{{dataset_folder}}/y_test_old.csv").squeeze("columns")

# Load new data
X_train_new = pd.read_csv(f"{{dataset_folder}}/X_train_new.csv")
y_train_new = pd.read_csv(f"{{dataset_folder}}/y_train_new.csv").squeeze("columns")
X_test_new = pd.read_csv(f"{{dataset_folder}}/X_test_new.csv")
y_test_new = pd.read_csv(f"{{dataset_folder}}/y_test_new.csv").squeeze("columns")

# Combine training data
X_train = pd.concat([X_train_old, X_train_new])
y_train = pd.concat([y_train_old, y_train_new])
"""

        # Add preprocessing if needed
        if numerical_features or categorical_features:
            code += """
# Define feature columns
"""
            if numerical_features:
                code += f"numerical_features = {numerical_features}\n"
            if categorical_features:
                code += f"categorical_features = {categorical_features}\n"
                
            code += """
# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
"""
            if numerical_features:
                code += """        ('num', StandardScaler(), numerical_features),
"""
            if categorical_features:
                code += """        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
"""
            code += """    ]
)

# Create and train pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', """ + model_init + """)
])
"""
        else:
            code += f"""
# Create and train model
model = {model_init}
"""
            
        code += """
model.fit(X_train, y_train)

# Evaluate on old distribution
old_dist_score = accuracy_score(y_test_old, model.predict(X_test_old))
print(f'Model evaluated on old distribution: {old_dist_score}')
model_new_score['on_old_data'] = float(old_dist_score)

# Evaluate on new distribution
new_dist_score = accuracy_score(y_test_new, model.predict(X_test_new))
print(f'Model evaluated on new distribution: {new_dist_score}')
model_new_score['on_new_data'] = float(new_dist_score)

# Save metrics
with open('automl_metrics.yaml', 'w') as f:
    yaml.dump({'model_new_score': model_new_score}, f)
"""
        return code


def run_automl_benchmark(training_code, dataset_description, metrics, max_iterations=3):
    """
    Run the AutoML benchmark on the given dataset.
    
    Args:
        training_code: Original model training code
        dataset_description: Dataset metadata
        metrics: Initial metrics dictionary
        max_iterations: Maximum number of iterations
        
    Returns:
        Dictionary with benchmark results
    """
    start_time = time.time()
    
    # Load data
    dataset_folder = "datasets/financial"
    X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
    X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
    y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
    y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
    
    X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
    y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
    X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
    y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
    
    # Create AutoML
    automl = DeterministicAutoML(max_iterations=max_iterations)
    
    # Get feature lists from dataset description
    numerical_features = dataset_description.get('NUMERICAL_FEATURES', [])
    categorical_features = dataset_description.get('CATEGORICAL_FEATURES', [])
    
    # Run AutoML
    result = automl.fit(
        X_train_old, y_train_old, X_test_old, y_test_old,
        X_train_new, y_train_new, X_test_new, y_test_new,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        dataset_description=dataset_description
    )
    
    return result
