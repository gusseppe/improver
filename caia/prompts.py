from langchain_core.prompts import ChatPromptTemplate
from typing import Dict
from caia.memory import SemanticMemory
from caia.representation import DatasetRepresentation
import textwrap
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


def escape_curly_braces(description):
    return description.replace("{", "{{").replace("}", "}}")

def prompt_measure_criticality() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            Monitoring Report Summary:
            - Feature drift: Income (15% shift), Credit Score (5% shift), Loan Amount (2% shift)
            - SHAP values changes:
              Income: 0.15 (prev) -> 0.18 (current), rank 1 -> 1
              Credit Score: 0.12 (prev) -> 0.11 (current), rank 2 -> 3
              Loan Amount: 0.10 (prev) -> 0.13 (current), rank 3 -> 2
            """,
            "output": """
            arguments_for_high_criticality:
              - Significant drift (15%) in Income feature indicates a notable shift in input data patterns
              - Income's SHAP value has increased, reinforcing its importance as the top feature
              - Loan Amount has become more important (rank 3 -> 2) and shows a slight drift

            arguments_for_low_criticality:
              - Credit Score drift (5%) is relatively minor
              - Despite rank changes, SHAP values for Credit Score and Loan Amount haven't changed drastically
              - No drift detected in features beyond the top 3

            criticality_score:
              score: 3
              reason: "The monitoring report indicates moderate concerns. The significant drift in the Income feature, coupled with its increased SHAP value, suggests that the model may need to be updated to better capture current income patterns. However, the stability of other features and minor changes in their SHAP values indicate that the model isn't critically misaligned. A score of 3 reflects the need for attention and potential model updates, but doesn't suggest an immediate crisis."
            """
        },
        {
            "input": """
            Monitoring Report Summary:
            - Feature drift: Loan Term (20% shift), Interest Rate (18% shift), Employment Length (12% shift)
            - SHAP values changes:
              Loan Term: 0.08 (prev) -> 0.14 (current), rank 5 -> 2
              Interest Rate: 0.11 (prev) -> 0.15 (current), rank 3 -> 1
              Employment Length: 0.09 (prev) -> 0.06 (current), rank 4 -> 6
            """,
            "output": """
            arguments_for_high_criticality:
              - Major drift in Loan Term (20%) and Interest Rate (18%) indicates substantial changes in key features
              - Significant changes in SHAP values and rankings for Loan Term and Interest Rate suggest a shift in feature importance
              - The model's understanding of feature relationships may be outdated given these substantial changes

            arguments_for_low_criticality:
              - Employment Length shows moderate drift but has decreased in importance according to SHAP values
              - The drift and SHAP changes are concentrated in a few features, suggesting others may still be reliable

            criticality_score:
              score: 4
              reason: "The monitoring report reveals high-concern issues that demand prompt attention. The substantial drift in Loan Term and Interest Rate, combined with their significantly increased importance (as shown by SHAP values), indicates that the model's understanding of these features is likely outdated. These changes suggest that the model may be making predictions based on outdated patterns, particularly for these critical features. A score of 4 reflects the urgent need for action, likely requiring model retraining or recalibration to address these shifts in feature distributions and importance."
            """
        }
    ]
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    system_prompt = """
    You are an expert in analyzing ML model monitoring reports, focusing on drift analysis and feature attribution. Your task is to review a monitoring report on a production ML model's performance and assess the level of concern it should raise.

    Context: You're analyzing a monitoring report for an ML model in production. This report contains information about feature drift and changes in feature importance (SHAP values).

    Objective: Analyze the monitoring report, provide clear arguments for why the findings should or should not be considered critical, and assign a criticality score. Your analysis should help in understanding the current state of the model based on drift and feature importance changes. Criticality Score Rubric:
        1 - No concern: No significant drift or SHAP value changes detected.
        2 - Low concern: Minor drift or SHAP value changes, likely not impacting model performance significantly.
        3 - Moderate concern: Noticeable drift or SHAP value changes that may require attention but don't pose immediate risk.
        4 - High concern: Significant drift or SHAP value changes that are likely impacting model performance and reliability.
        5 - Critical concern: Severe drift or SHAP value changes detected that require immediate action to prevent model failure.

    Style: Use clear, concise language focusing on technical details, as is common in developer communication. Be straightforward and to the point, using technical terms when relevant.

    Tone: Maintain a technical and objective tone, presenting facts and analysis without emotional coloring. Your response should convey expertise and inspire confidence in your assessment.

    Audience: Your analysis is for a product owner or data scientist who needs to understand the implications of the monitoring report. Balance technical accuracy with accessibility, considering that they may not have deep ML expertise but understand the business implications.

    Response Format: Format your response as YAML, structured as follows:
                                    
    arguments_for_high_criticality:
      - [Argument 1]
      - [Argument 2]
      ...
    arguments_for_low_criticality:
      - [Argument 1]
      - [Argument 2]
      ...
    criticality_score:
      score: [1-5]
      reason: "[Detailed explanation for the score]"

    Only answer the arguments_for_high_criticality, arguments_for_low_criticality, and criticality_score sections. Do not include any other information in your response.

    
    """
    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    return final_prompt


def prompt_measure_model_retraining_speed() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            num_samples: 500
            num_features: 3
            training_code: |
              from sklearn.linear_model import LogisticRegression
              model = LogisticRegression()
              model.fit(X_train, y_train)
            """,
            "output": """
            score: 1
            reason: "Very fast retraining expected. Small dataset (500 samples, which is < 100,000) with few features (3, which is < 20) and a simple model (Logistic Regression). Estimated training time: < 5 seconds."
            """
        },
        {
            "input": """
            num_samples: 50000
            num_features: 25
            training_code: |
              from sklearn.ensemble import RandomForestClassifier
              model = RandomForestClassifier(n_estimators=100)
              model.fit(X_train, y_train)
            """,
            "output": """
            score: 2
            reason: "Fast retraining expected. Small dataset (50,000 samples, which is < 100,000) but with moderate number of features (25, which is >= 20). Using a parallel-friendly model (Random Forest, 100 estimators). Modern hardware can efficiently handle this task. Estimated training time: 5-30 seconds."
            """
        },
        {
            "input": """
            num_samples: 1000000
            num_features: 100
            training_code: |
              from xgboost import XGBClassifier
              from sklearn.preprocessing import StandardScaler
              from sklearn.pipeline import Pipeline
              
              pipeline = Pipeline([
                  ('scaler', StandardScaler()),
                  ('model', XGBClassifier(n_estimators=1000, max_depth=10))
              ])
              pipeline.fit(X_train, y_train)
            """,
            "output": """
            score: 3
            reason: "Moderate retraining time expected. Large dataset (1,000,000 samples, which is >= 100,000) with many features (100, which is >= 20). Using XGBoost which is highly optimized for parallel processing. StandardScaler preprocessing adds some overhead but is efficient. Modern hardware and XGBoost's optimizations can handle this task relatively well. Estimated training time: 30-60 seconds."
            """
        }
    ]
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    system_prompt = """
    You are an expert in machine learning model training and optimization. Your task is to analyze the given training set information and training code to assess the expected speed of model retraining.

    Context: The user wants to check the feasibility of retraining their ML model based on provided training data and code details.

    Objective: Analyze the given information and assign a score from 1 to 5 indicating the expected retraining speed, where:
    1 - Very Fast (< 5 seconds)
    2 - Fast (5-30 seconds)
    3 - Moderate (30-60 seconds)
    4 - Slow (1-5 minutes)
    5 - Very Slow (> 5 minutes)
    Provide a detailed explanation for your assigned score, including an estimated training time range.

    The score should be an integer between 1 and 5, and the reason should provide a comprehensive explanation for the assigned score. Be optimistic, assuming the user has recent hardware with powerful core speed.

    Base your assessment on:
    1. Number of samples (consider < 100,000 as small)
    2. Number of features (consider < 20 as small)
    3. Complexity of the training code (model type, preprocessing steps, etc.)

    When assessing, explicitly mention whether the number of samples is considered small (< 100,000) or large (>= 100,000), and whether the number of features is considered small (< 20) or large (>= 20).

    Style: Provide a technical analysis focusing on how dataset characteristics and model complexity interact to affect training speed. Be specific and quantitative where possible.

    Tone: Maintain a professional and objective tone, suitable for a technical audience. Use precise terminology and quantitative reasoning.

    Audience: Your response is for product owners or data scientists who need to understand if retraining a model will be fast or slow, why, and how long it might take.

    Response Format: Format your response as YAML output with the following structure:

    score: [1-5]
    reason: [detailed reason including estimated training time]

    Only answer the score and reason sections. Do not include any other information in your response.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt


def prompt_stats_data() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            old:
              description:
                NUMERICAL_FEATURES: [Age, Income]
                CATEGORICAL_FEATURES: [Home_Ownership]
                COLUMN_VALUES:
                  Home_Ownership: [0: Rent, 1: Own]
              x_train: data/ref/X.csv
              y_train: data/ref/y.csv
            new:
              description:
                NUMERICAL_FEATURES: [Age, Income]
                CATEGORICAL_FEATURES: [Home_Ownership]
              x_train: data/new/X.csv
              y_train: data/new/y.csv
            """,
            "output": """
            stats_code: |
                import pandas as pd
                import numpy as np
                import yaml

                def calculate_drift(ref_stat, new_stat):
                    return round((new_stat - ref_stat) / ref_stat, 3)

                def analyze_dataset(X_path, y_path, description):
                    X = pd.read_csv(X_path)
                    y = pd.read_csv(y_path).squeeze()

                    stats = {
                        'class_distribution': y.value_counts(normalize=True).round(3).to_dict(),
                        'numerical': {feat: {
                            'mean': round(X[feat].mean(), 2),
                            'std': round(X[feat].std(), 2),
                            'min': X[feat].min(),
                            'max': X[feat].max()
                        } for feat in description['NUMERICAL_FEATURES']},
                        'categorical': {feat: {
                            'distribution': X[feat].replace(
                                description.get('COLUMN_VALUES', {}).get(feat, {})
                            ).value_counts(normalize=True).round(3).to_dict()
                        } for feat in description['CATEGORICAL_FEATURES']},
                        'missing': X.isna().mean().round(3).to_dict()
                    }
                    return stats

                # Analyze datasets
                ref_stats = analyze_dataset("data/ref/X.csv", "data/ref/y.csv", old['description'])
                new_stats = analyze_dataset("data/new/X.csv", "data/new/y.csv", new['description'])

                # Calculate drift metrics
                drift_metrics = {
                    'class_distribution': calculate_drift(
                        ref_stats['class_distribution'].get(1, 0),
                        new_stats['class_distribution'].get(1, 0)
                    ),
                    'numerical': {
                        feat: calculate_drift(
                            ref_stats['numerical'][feat]['mean'],
                            new_stats['numerical'][feat]['mean']
                        ) for feat in ref_stats['numerical']
                    },
                    'categorical': {
                        feat: calculate_drift(
                            ref_stats['categorical'][feat]['distribution'].get(1, 0),
                            new_stats['categorical'][feat]['distribution'].get(1, 0)
                        ) for feat in ref_stats['categorical']
                    }
                }

                # Save results
                with open('dataset_stats.yaml', 'w') as f:
                    yaml.dump({
                        'old_stats': ref_stats,
                        'new_stats': new_stats,
                        'drift_metrics': drift_metrics
                    }, f)
            """
        }
    ]

    system_prompt = """
    Generate Python code to analyze and compare two datasets (old and new). The code must:
    1. Compute statistics for numerical features (mean, std, min, max).
    2. Analyze distributions for categorical features.
    3. Calculate missing value percentages.
    4. Compute drift metrics between datasets.
    5. Save results in 'dataset_stats.yaml'.

    Response Format:

    stats_code: |
        [COMPLETE STATS CODE]

    Only provide the YAML-formatted code output. Do not include any other explanation or commentary.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages([
                ("human", "{input}"),
                ("ai", "{output}")
            ]),
            examples=examples
        ),
        ("human", "{input}")
    ])

    return final_prompt



def prompt_generate_retraining_code_with_insights() -> ChatPromptTemplate:
    examples = [
        {
            "input": '''
            old_training_code: |
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier
                # load the old data
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                model_old = RandomForestClassifier(random_state=42)
                model_old.fit(X_train_old, y_train_old)
                # Test the model on the initial test set
                initial_accuracy = model_old.score(X_test_old, y_test_old)
                print(f'Model trained and evaluated on the old distribution: {initial_accuracy}')
            
            improved_model_code: |
                import pandas as pd
                from sklearn.ensemble import GradientBoostingClassifier
                
                # load the old data
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                model = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42
                )
                
                model.fit(X_train_old, y_train_old)
                
                # Test the model on the initial test set
                initial_accuracy = model.score(X_test_old, y_test_old)
                
                print(f'Model trained and evaluated on the old distribution: {initial_accuracy}')
            ''',
            "output": '''
            new_training_code: |
                import yaml
                import pandas as pd
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.metrics import accuracy_score
                
                # Initialize metrics dictionaries
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                model_old_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                try:
                    # Load data from specified folder
                    dataset_folder = "datasets/financial"
                    X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                    X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                    y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                    y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                    
                    # Train improved model on old data only
                    model_old = GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        random_state=42
                    )
                    model_old.fit(X_train_old, y_train_old)
                    
                    # Evaluate improved model on old test set (ONLY test data)
                    old_score_old = accuracy_score(y_test_old, model_old.predict(X_test_old))
                    print(f'Old model trained and evaluated on the old distribution: {old_score_old}')
                    model_old_score['on_old_data'] = float(old_score_old)
                    
                    # Load new data
                    X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                    X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                    y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                    y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                    
                    # Evaluate improved model on new test set (ONLY test data)
                    old_score_new = accuracy_score(y_test_new, model_old.predict(X_test_new))
                    print(f'Old model evaluated on the new distribution: {old_score_new}')
                    model_old_score['on_new_data'] = float(old_score_new)
                    
                    # Save old model metrics
                    with open('old_metrics.yaml', 'w') as f:
                        yaml.dump({'model_old_score': model_old_score}, f)
                    
                    print("\\nTraining new model on combined data...")
                    
                    # Combine training datasets for retraining
                    X_train = pd.concat([X_train_old, X_train_new])
                    y_train = pd.concat([y_train_old, y_train_new])
                    
                    # Create and train new model with improved configuration
                    model_new = GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4, 
                        subsample=0.8,
                        random_state=42
                    )
                    model_new.fit(X_train, y_train)
                    
                    # Evaluate new model on old test set (ONLY test data)
                    new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                    print(f'New model trained and evaluated on old distribution: {new_score_old}')
                    model_new_score['on_old_data'] = float(new_score_old)
                    
                    # Evaluate new model on new test set (ONLY test data)
                    new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                    print(f'New model evaluated on new distribution: {new_score_new}')
                    model_new_score['on_new_data'] = float(new_score_new)
                    
                    # Save new model metrics
                    with open('fast_graph_metrics.yaml', 'w') as f:
                        yaml.dump({'model_new_score': model_new_score}, f)
                        
                except FileNotFoundError as e:
                    print(f"Required data file not found: {str(e)}")
                    print("Ensure all train/test files for old and new data exist.")
                except Exception as e:
                    print(f"Error during model training/evaluation: {str(e)}")
            '''
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    system_prompt = """
    You are an expert machine learning engineer specializing in generating fair and methodologically sound retraining code.
    
    Context: You have been given:
    1. Original training code
    2. Improved model code from comprehensive ML analysis
    3. Metrics and insights from model improvement process
    
    CRITICAL EVALUATION REQUIREMENTS - YOUR CODE WILL BE REJECTED IF THESE ARE VIOLATED:
    1. NEVER use cross-validation on test data (X_test_old, X_test_new)
    2. NEVER evaluate on training data for final metrics
    3. NEVER fit preprocessing (scalers, etc.) separately on test data
    4. ONLY evaluate using: accuracy_score(y_test, model.predict(X_test))
    5. Ensure proper train/test separation throughout the pipeline
    6. Use accuracy_score() instead of model.score() for consistency
    7. Train/validation splits should ONLY be applied to training data, never test data
    
    VALID EVALUATION PATTERN EXAMPLE:
    ```python
    # CORRECT: Evaluate on separate test sets using accuracy_score
    old_score_old = accuracy_score(y_test_old, model_old.predict(X_test_old))
    new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
    
    # CORRECT: Use internal validation for model tuning if needed
    model = GradientBoostingClassifier(validation_fraction=0.1, n_iter_no_change=10)
    ```
    
    INVALID PATTERNS TO AVOID:
    ```python
    # WRONG: Cross-validation on test data
    score = cross_val_score(model, X_test_old, y_test_old, cv=5)
    
    # WRONG: Evaluation on training data
    score = accuracy_score(y_train, model.predict(X_train))
    
    # WRONG: Using model.score() inconsistently
    score = model.score(X_test_old, y_test_old)  # Use accuracy_score instead
    ```
    
    Objective: Create new retraining code (new_training_code) that:
    1. **First evaluates the improved model on both distributions**:
       - Uses the improved model configuration from analysis
       - Trains ONLY on old training data (X_train_old, y_train_old)
       - Evaluates on old test set using accuracy_score (X_test_old, y_test_old)
       - Evaluates on new test set using accuracy_score (X_test_new, y_test_new)
       - Saves model metrics to 'old_metrics.yaml'
    
    2. **Then trains a new model on combined data and evaluates it**:
       - Uses the same improved model architecture and parameters
       - Trains on combined training dataset (X_train_old + X_train_new)
       - Evaluates on old test set using accuracy_score (X_test_old, y_test_old)
       - Evaluates on new test set using accuracy_score (X_test_new, y_test_new)
       - Saves metrics to 'fast_graph_metrics.yaml'
    
    3. **Includes proper error handling and validation**:
       - Add try-catch blocks for file loading and model training
       - Provide informative error messages
       - Ensure all required data files are loaded
    
    4. **Prints performance metrics at each step** with clear, descriptive messages
    
    Critical Requirements:
    1. Use consistent evaluation methodology with accuracy_score()
    2. Maintain proper metrics format:
       model_new_score:
           on_new_data: [score]
           on_old_data: [score]
       model_old_score:
           on_new_data: [score]
           on_old_data: [score]
    3. MANDATORY: Evaluate ONLY on dedicated test sets
    4. Preserve dataset folder paths from input code
    5. Include comprehensive error handling
    6. Follow the exact workflow: improved model evaluation → retraining → final evaluation
    
    Your code must leverage the insights and model improvements from the analysis while ensuring robust, fair evaluation across distributions.
    
    Style Guidelines:
    1. Clear, well-structured Python code
    2. Descriptive variable names and comments
    3. Consistent evaluation methodology
    4. Proper error handling and validation
    5. Clean metrics saving and file organization
    
    Response Format: Format your response as YAML output with the following structure:
    new_training_code: |
      [COMPLETE RETRAINING CODE WITH FAIR EVALUATION]
    
    The implementation must:
    1. Follow strict train/test separation principles
    2. Use only test sets for final evaluation
    3. Include proper error handling and validation
    4. Maintain consistent evaluation methodology
    5. Leverage improved model insights appropriately
    
    REMEMBER: Any code that violates fair evaluation principles will be automatically rejected.
    Only evaluate on dedicated test sets that were never used for training or validation.
    
    Only provide the YAML-formatted code output. Do not include any other explanation or commentary.
    """
    system_prompt = textwrap.dedent(system_prompt).strip()
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    return final_prompt


def prompt_generate_retraining_code() -> ChatPromptTemplate:
    examples = [
        {
            "input": '''
            old_training_code: |
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier

                # load the old data
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")

                model_old = RandomForestClassifier(random_state=42)

                model_old.fit(X_train_old, y_train_old)

                # Test the model on the initial test set
                initial_accuracy = model_old.score(X_test_old, y_test_old)

                print(f'Model trained and evaluated on the old distribution: {initial_accuracy}')
            ''',
            "output": '''
            new_training_code: |
                import yaml
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier

                # Initialize metrics dictionaries
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                model_old_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }

                # load the old data
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")

                # Train and evaluate old model
                model_old = RandomForestClassifier(random_state=42)
                model_old.fit(X_train_old, y_train_old)

                # Test old model on old test set
                old_score_old = model_old.score(X_test_old, y_test_old)
                print(f'Old model trained and evaluated on the old distribution: {old_score_old}')
                model_old_score['on_old_data'] = float(old_score_old)

                # Test old model on new test set
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                old_score_new = model_old.score(X_test_new, y_test_new)
                print(f'Old model evaluated on the new distribution: {old_score_new}')
                model_old_score['on_new_data'] = float(old_score_new)

                # Save old model metrics
                with open('old_metrics.yaml', 'w') as f:
                    yaml.dump({'model_old_score': model_old_score}, f)

                print("\\nTraining new model on combined data...")

                # load and combine new training data
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])

                # Train new model on combined dataset
                model_new = RandomForestClassifier(random_state=42)
                model_new.fit(X_train, y_train)

                # Test new model on old test set
                new_score_old = model_new.score(X_test_old, y_test_old)
                print(f'New model trained and evaluated on old distribution: {new_score_old}')
                model_new_score['on_old_data'] = float(new_score_old)

                # Test new model on new test set
                new_score_new = model_new.score(X_test_new, y_test_new)
                print(f'New model evaluated on new distribution: {new_score_new}')
                model_new_score['on_new_data'] = float(new_score_new)

                # Save new model metrics
                with open('fast_graph_metrics.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            '''
        }
    ]

    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    system_prompt = """
    You are an expert machine learning engineer. You have to rewrite the given training code to obtain a retraining code.

    Context: Given a old training code and new data loading code, generate a new training code that retrains the model.

    Objective: Create a new training code (new_training_code) that:
    1. First evaluates the old model on both distributions:
       - Trains on old data
       - Tests on old test set 
       - Tests on new (drifted) test set
       - Saves old model metrics to 'old_metrics.yaml' with structure:
         model_old_score:
           on_new_data: [score on new data]
           on_old_data: [score on old data]
    2. Then trains a new model on combined data and evaluates it:
       - Uses the same ML model and parameters
       - Trains on combined dataset (old + new data)
       - Tests on old test set
       - Tests on new (drifted) test set
       - Saves new model metrics to 'fast_graph_metrics.yaml' with structure:
         model_new_score:
           on_new_data: [score on new data]
           on_old_data: [score on old data]
    3. Prints performance metrics and data shapes at each step

    Do not use BaggingClassifier, use a different approach.

    Style: Provide clear, well-structured Python code that maintains the same model architecture and parameters.

    Response Format: Format your response as YAML output with the following structure:

    new_training_code: |
      [NEW TRAINING/EVALUATING CODE HERE]

    Only provide the YAML-formatted code output. Do not include any other explanation or commentary.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt



def prompt_execute_and_fix_retraining_code() -> ChatPromptTemplate:
    
    # example_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("human", "{input}"),
    #         ("ai", "{output}"),
    #     ]
    # )
    
    # few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     example_prompt=example_prompt,
    #     examples=examples,
    # )

    system_prompt = """
    You are an expert machine learning engineer tasked with fixing errors in ML code.

    Context: Given a piece of ML code with an error output, your job is to identify and fix the error(s) in the code.

    Objective: Create a fixed version of the code (fixed_code) that:
    1. Addresses the specific error(s) mentioned in the error output by modifying the code.
    2. Maintains the original functionality and structure of the code as much as possible.
    4. Includes clear comments explaining the fixes made.
    5. Ensures the code runs without errors and performs the intended ML task.
    6. Do not use BaggingClassifier, replace it with a different approach.

    Style: Provide a technical, precise, and well-structured code fix.

    Audience: Your response is for a data scientist who needs to run the fixed code and understand the changes made.

    Response Format: Format your response as YAML output with the following structure:

    fixed_code: |
      [FIXED ML CODE HERE]

    Only provide the YAML-formatted code output. Do not include any other explanation or commentary outside the code block.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            # few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt




def prompt_fix_updating_code():
    system_prompt = """
    You are an expert machine learning engineer tasked with fixing and improving model code based on evaluation results.

    Context: You have been provided with the current model code and its evaluation results.

    Objective: Analyze the evaluation results and modify the model code to address any issues or improve performance. Your updated code should:
    1. Address any problems identified in the evaluation
    2. Implement changes that could potentially improve model performance
    3. Be clear, well-structured, and properly commented
    4. Be complete and runnable

    Style: Provide clear, well-structured, and commented Python code.

    Audience: Your code will be executed by a data scientist to train and evaluate the improved model.

    Response Format: Provide only the Python code, properly formatted and commented. Include a brief comment at the top summarizing the changes made.
    """

    human_prompt = """
    Based on the following evaluation results, please improve the model code:

    Current Model Code: {current_code}
    Evaluation Results: {evaluation_result}

    Provide the improved model code below:
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])



from langchain_core.prompts import ChatPromptTemplate
from typing import Dict
import textwrap


def prompt_distill_memories() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            execution_output: |
                Old model trained and evaluated on the old distribution: 0.913
                Old model evaluated on the new distribution: 0.717
                
                Training new model on combined data...
                New model trained and evaluated on old distribution: 0.907
                New model evaluated on new distribution: 0.800
            model_code: |
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
            """,
            "output": """
            insights:
              performance_analysis:
                old_model:
                  - Strong baseline on old distribution (0.913)
                  - Significant drop on new distribution (0.717)
                  - Performance gap of 19.6% between distributions
                new_model:
                  - Maintained strong old distribution performance (0.907)
                  - Improved new distribution handling (0.800)
                  - Reduced gap to 10.7% between distributions
                key_metrics:
                  - Improvement of 8.3% on new distribution
                  - Minor decrease of 0.6% on old distribution
                  - Overall better distribution balance
              
              model_limitations:
                - Basic RandomForest with default parameters
                - No explicit drift handling mechanisms
                - Default n_estimators may be insufficient
                - Unlimited tree depth potential overfitting
                - No class balancing consideration
              
              hyperparameter_recommendations:
                primary_changes:
                  n_estimators: 500
                  max_depth: 15
                  min_samples_split: 10
                  class_weight: 'balanced'
                  max_features: 'sqrt'
                  bootstrap: True
                
              alternative_models:
                gradient_boosting:
                  rationale: "Better handling of distribution shifts"
                  suggested_config:
                    - model: "GradientBoostingClassifier"
                    - n_estimators: 300
                    - learning_rate: 0.1
                    - max_depth: 5
                    - subsample: 0.8
              
              improvement_priority:
                1: "Optimize RandomForest parameters"
                2: "Consider GradientBoosting if needed"
                3: "Implement robust validation strategy"
              
              expected_impacts:
                - Further reduction in distribution gap
                - More robust generalization
                - Maintained old distribution performance
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    system_prompt = """
    You are an expert ML engineer analyzing model performance using sklearn components.

    Context: You have access to:
    1. Performance metrics for both models on old and new distributions
    2. Base model code and configuration

    Objective: Analyze the model performance and provide sklearn-specific insights about:
    1. Detailed performance comparison between old and new models
    2. Quantified gaps and improvements
    3. Current model limitations
    4. Specific sklearn parameter recommendations

    Analysis Requirements:
    1. Calculate performance gaps between distributions
    2. Identify key improvements achieved
    3. Suggest specific parameter optimizations
    4. Focus on sklearn-native solutions

    Response Format: Format your response as YAML with:

    insights:
      performance_analysis:
        old_model:
          - [Performance metrics and gaps]
        new_model:
          - [Performance metrics and gaps]
        key_metrics:
          - [Key improvements and changes]
      
      model_limitations:
        - [Current parameter limitations]
      
      hyperparameter_recommendations:
        primary_changes:
          [Specific parameter adjustments]
      
      alternative_models:
        [sklearn_model]:
          rationale: [Reasoning]
          suggested_config: [Parameters]
      
      improvement_priority:
        [Ordered list of next steps]

      expected_impacts:
        - [Expected outcomes]

    Only provide the YAML-formatted output. No additional commentary.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    return final_prompt

def prompt_generate_tiny_change() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_code: |
                new_training_code: |
                    import pandas as pd
                    import yaml
                    from sklearn.ensemble import RandomForestClassifier
                    
                    metrics = {
                        'model_old': {},
                        'model_new': {},
                        'difference_score_averages': {}
                    }
                    
                    # Train old model
                    model_old = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
                    model_old.fit(X_train_old, y_train_old)
                    
            execution_results:
                output: |
                    Model trained and evaluated on the old distribution: 0.902
                    old model evaluated on the new distribution: 0.617
                    Average score of old model: 0.759
                    Training new model on combined data...
                    New model evaluated on old distribution: 0.902
                    New model evaluated on new distribution: 0.642
                    Average score of new model: 0.772
                    Score difference: 0.013
                old_score: 0.902
                new_distribution_score: 0.617
                score_difference: 0.013
                success: true
                
            distilled_insights: |
                insights:
                    performance_gaps:
                        - Large gap between old and new distribution
                        - Model not generalizing well
                    model_characteristics:
                        - Basic RandomForest configuration
                        - Missing robustness mechanisms
                    critical_improvements_needed:
                        - Increase ensemble size
                        - Control tree depth
                        - Add leaf node controls
                        
            criticality_analysis: |
                arguments_for_high_criticality:
                    - Significant drift in multiple features
                    - Changes in feature importance
                criticality_score:
                    score: 4
                    
            iteration_count: 1
            
            previous_changes: []
            """,
            "output": """
            new_training_code: |
                import pandas as pd
                import yaml
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score
                
                metrics = {
                    'model_old': {},
                    'model_new': {},
                    'difference_score_averages': {}
                }
                
                # Load old data
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Configure and train old model with enhanced robustness
                model_old = RandomForestClassifier(
                    n_estimators=500,           # Increased for stability
                    max_depth=15,               # Prevent overfitting
                    min_samples_split=20,       # Conservative splits
                    min_samples_leaf=10,        # Stable leaves
                    max_features='sqrt',        # Feature randomization
                    class_weight='balanced',    # Handle imbalance
                    bootstrap=True,             # Enable bootstrapping
                    random_state=42
                )
                model_old.fit(X_train_old, y_train_old)
                
                # Evaluate old model
                y_pred_old = model_old.predict(X_test_old)
                ref_score_old = accuracy_score(y_test_old, y_pred_old)
                print(f'Model trained and evaluated on the old distribution: {ref_score_old}')
                metrics['model_old']['score_old_data'] = float(ref_score_old)
                
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                y_pred_new = model_old.predict(X_test_new)
                ref_score_new = accuracy_score(y_test_new, y_pred_new)
                print(f'old model evaluated on the new distribution: {ref_score_new}')
                metrics['model_old']['score_new_data'] = float(ref_score_new)
                
                ref_score_average = (ref_score_old + ref_score_new) / 2
                print(f'Average score of old model: {ref_score_average}')
                metrics['model_old']['score_average'] = float(ref_score_average)
                
                print("\\nTraining new model on combined data...")
                
                # Load and combine new data
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                # Train new model with same robust configuration
                model_new = RandomForestClassifier(
                    n_estimators=500,           # Increased for stability
                    max_depth=15,               # Prevent overfitting
                    min_samples_split=20,       # Conservative splits
                    min_samples_leaf=10,        # Stable leaves
                    max_features='sqrt',        # Feature randomization
                    class_weight='balanced',    # Handle imbalance
                    bootstrap=True,             # Enable bootstrapping
                    random_state=42
                )
                model_new.fit(X_train, y_train)
                
                # Evaluate new model
                y_pred_old_new = model_new.predict(X_test_old)
                new_score_old = accuracy_score(y_test_old, y_pred_old_new)
                print(f'New model evaluated on old distribution: {new_score_old}')
                metrics['model_new']['score_old_data'] = float(new_score_old)
                
                y_pred_new_new = model_new.predict(X_test_new)
                new_score_new = accuracy_score(y_test_new, y_pred_new_new)
                print(f'New model evaluated on new distribution: {new_score_new}')
                metrics['model_new']['score_new_data'] = float(new_score_new)
                
                new_score_average = (new_score_old + new_score_new) / 2
                print(f'Average score of new model: {new_score_average}')
                metrics['model_new']['score_average'] = float(new_score_average)
                
                score_difference = new_score_average - ref_score_average
                print(f'\\nScore difference: {score_difference}')
                metrics['difference_score_averages']['score_average'] = float(score_difference)
                
                with open('retraining_metrics.yaml', 'w') as f:
                    yaml.dump(metrics, f)
            
            changes_made:
              - Increased n_estimators from 100 to 500
              - Added max_depth=15 for overfitting control
              - Added min_samples_split=20 for conservative splits
              - Added min_samples_leaf=10 for leaf stability
              - Set max_features='sqrt' for feature randomization
              - Added class_weight='balanced'
              - Enabled bootstrap sampling
            
            rationale: |
                First iteration focuses on fundamental robustness improvements:
                1. Larger ensemble (500 trees) to improve stability and handle drift
                2. Tree depth control (15) to prevent overfitting to old data
                3. Conservative split and leaf parameters to build more stable trees
                4. Feature randomization and class balancing to handle distribution changes
                These changes aim to create a more robust model that should generalize 
                better across distributions while maintaining good performance on the
                old dataset.
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    system_prompt = """
    You are an expert ML engineer tasked with iteratively improving a RandomForest model's performance through parameter tuning.

    Context: You have:
    1. Current model code and configuration
    2. Detailed execution results
    3. Performance metrics
    4. Previous changes (if any)
    5. Criticality analysis
    
    Objective: Generate a new version of the training code that:
    1. Uses more appropriate RandomForestClassifier parameters
    2. Maintains exact code structure and data handling
    3. Makes meaningful parameter changes based on results
    4. Improves model robustness and generalization
    
    Rules:
    1. DO NOT change any other code structure 
    2. ONLY modify RandomForestClassifier parameters
    3. DO NOT add preprocessing or feature engineering
    4. Each iteration should try significantly different parameters
    5. Parameters should progress logically based on results
    6. If you don't see significant improvement, change the model architecture
    
    Parameter Progression Guidelines:
    - Iteration 1: Focus on fundamental robustness (n_estimators, max_depth)
    - Iteration 2: Tune tree construction (min_samples_split, min_samples_leaf)
    - Iteration 3: Adjust feature selection (max_features, max_samples)
    - Iteration 4: Fine-tune previous best parameters
    - Iteration 5: Make more aggressive changes if needed
    
    Response Format: Format your response as YAML with:
    
    new_training_code: |
        [COMPLETE UPDATED TRAINING CODE]
    changes_made:
        - [List each parameter change]
    rationale: |
        [Detailed explanation of changes]

    Each response MUST:
    1. Include complete, runnable code
    2. List ALL parameter changes
    3. Explain reasoning for changes
    4. Consider previous results
    5. Make meaningful parameter adjustments

    DO NOT suggest preprocessing changes or new features.
    FOCUS ONLY on RandomForestClassifier parameters.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    return final_prompt

def prompt_evaluate_change() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_code: |
                from sklearn.ensemble import StackingClassifier
                
                model_new = StackingClassifier(
                    estimators=[
                        ('gb', GradientBoostingClassifier(n_estimators=200)),
                        ('rf', RandomForestClassifier(n_estimators=200))
                    ],
                    final_estimator=LogisticRegression(),
                    cv=5
                )
            
            execution_output: |
                New model trained and evaluated on old distribution: 0.907
                New model evaluated on new distribution: 0.800
            
            current_metrics:
                on_old_data: 0.907
                on_new_data: 0.800
            
            previous_metrics:
                on_old_data: 0.913
                on_new_data: 0.717
                
            strategy_type: "ensemble_method"
            
            improvements:
                old_distribution: -0.006
                new_distribution: 0.083
            """,
            "output": """
            evaluation:
              performance_metrics:
                distribution_gaps:
                  previous_gap: 0.196  # 0.913 - 0.717
                  current_gap: 0.107   # 0.907 - 0.800
                  gap_reduction: 0.089  # 0.196 - 0.107
                improvements:
                  old_distribution: -0.006  # Slight regression
                  new_distribution: 0.083   # Significant improvement
                relative_changes:
                  old_distribution_percent: -0.66%
                  new_distribution_percent: 11.58%
              
              analysis:
                - "Significant improvement on new distribution (+8.3%)"
                - "Minimal regression on old distribution (-0.6%)"
                - "Distribution gap reduced by 8.9 percentage points"
                - "Better balance between distributions achieved"
                - "Trade-off between distributions is acceptable"
              
              risk_assessment:
                - "10.7% remaining performance gap"
                - "Small regression on old distribution is within tolerance"
                - "New distribution improvement outweighs minor regression"
                - "Ensemble shows good adaptation capability"
              
              strategy_effectiveness:
                approach: "ensemble_method"
                strengths:
                  - "Successfully combines multiple model strengths"
                  - "Better handles distribution shift"
                  - "Maintains most old distribution performance"
                limitations:
                  - "Slight regression on old distribution"
                  - "Added model complexity"
              
              recommendation:
                action: "accept"
                confidence: "high"
                reasoning: "Strong improvement on new distribution with minimal old distribution impact"
              
              next_steps:
                - "Consider hyperparameter_tuning to fine-tune ensemble weights"
                - "Try model_selection for additional base estimators"
                - "Explore ensemble_method with different stacking strategies"
            """
        }
    ]
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    system_prompt = """
    You are an expert ML engineer evaluating model improvements. Your goal is to assess changes and recommend next steps.
    
    CRITICAL EVALUATION REQUIREMENTS:
    1. ONLY accept evaluations based on separate test sets (X_test_old, X_test_new)
    2. REJECT any code that uses cross-validation on test data
    3. REJECT any code that evaluates on training data for final metrics
    4. REJECT any code with data leakage (e.g., fitting scalers separately on test data)
    5. REQUIRE proper train/test separation throughout the pipeline
    
    INVALID EVALUATION PATTERNS TO REJECT:
    - cross_val_score(model, X_test, y_test) - CV should NEVER be used on test data
    - model.score(X_train, y_train) for final evaluation - training data should not be used for final metrics
    - Fitting preprocessing on test data separately (e.g., scaler.fit_transform(X_test))
    - Using validation sets created from already-split training data
    - Any evaluation methodology that violates train/test separation
    
    VALID EVALUATION PATTERN:
    - model.score(X_test_old, y_test_old) and model.score(X_test_new, y_test_new)
    - Using dedicated, separate test sets that were never seen during training
    - Proper preprocessing fitted only on training data and applied to test data
    
    Context: You have:
    1. Current and previous model metrics
    2. Applied strategy type
    3. Calculated improvements
    4. Model code changes
    
    Objective: Provide a comprehensive evaluation that:
    1. First validates evaluation methodology is fair and uses proper test sets
    2. Analyzes performance changes in detail
    3. Assesses distribution adaptation
    4. Evaluates strategy effectiveness
    5. Recommends clear next steps
    
    Evaluation Focus:
    1. Methodology validation:
       - Confirm evaluation uses separate test sets only
       - Check for data leakage or invalid evaluation practices
       - Verify proper train/test separation
    2. Distribution gap analysis:
       - Previous gap (old vs new distribution)
       - Current gap
       - Gap reduction
    3. Performance trade-offs:
       - Old distribution maintenance
       - New distribution improvement
       - Acceptable regression thresholds
    4. Strategy effectiveness:
       - Approach strengths
       - Limitations
       - Risk assessment
    5. Next steps based on available strategies:
       - model_selection
       - hyperparameter_tuning
       - ensemble_method
    
    Success Criteria:
    1. MANDATORY: Fair evaluation using proper test sets only
    2. Significant improvement on new distribution (>2%)
    3. Minimal regression on old distribution (<1%)
    4. Overall distribution gap reduction
    5. Clear strategy effectiveness
    
    Response Format: Format your response as YAML with:
    evaluation:
      methodology_check:
        valid_evaluation: [true/false]
        issues_found: [list any methodology problems]
      
      performance_metrics:
        distribution_gaps:
          [gap analysis]
        improvements:
          [improvement metrics]
        relative_changes:
          [percentage changes]
      
      analysis:
        - [Key findings]
      
      risk_assessment:
        - [Potential risks]
      
      strategy_effectiveness:
        approach: [strategy_type]
        strengths: [list]
        limitations: [list]
      
      recommendation:
        action: ["accept" or "reject"]
        confidence: ["high", "medium", or "low"]
        reasoning: [explanation]
      
      next_steps:
        - [ONLY from: model_selection, hyperparameter_tuning, ensemble_method]
    
    IMPORTANT: If methodology_check.valid_evaluation is false, automatically set recommendation.action to "reject" regardless of performance improvements.
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    system_prompt = textwrap.dedent(system_prompt).strip()
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    return final_prompt


def prompt_fix_code() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_code: |
                import pandas as pd
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.metrics import accuracy_score
                import yaml
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Initialize metrics dictionaries
                model_new_score = {
                    'on_new_data': 0.0
                }  # Missing on_old_data
                
                model_old_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                model = GradientBoostingClassifier(n_estimators=150)
                model.fit(X_train_old)  # Missing y_train
            
            error_output: |
                TypeError: fit() missing 1 required positional argument: 'y'
            """,
            "output": """
            fixed_code: |
                import pandas as pd
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.metrics import accuracy_score
                import yaml
                
                # Initialize metrics dictionary with proper structure
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                try:
                    # Load data from specified folder
                    dataset_folder = "datasets/financial"
                    X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                    X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                    y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                    y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                    
                    # Load new data for retraining
                    X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                    y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                    X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                    y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                    
                    # Combine training data
                    X_train = pd.concat([X_train_old, X_train_new])
                    y_train = pd.concat([y_train_old, y_train_new])
                    
                    # Train model with proper arguments
                    model_new = GradientBoostingClassifier(n_estimators=150, random_state=42)
                    model_new.fit(X_train, y_train)  # Fixed: Added y_train argument
                    
                    # Evaluate new model on old test set (ONLY test data)
                    new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                    print(f'New model trained and evaluated on old distribution: {new_score_old}')
                    model_new_score['on_old_data'] = float(new_score_old)
                    
                    # Evaluate new model on new test set (ONLY test data)
                    new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                    print(f'New model evaluated on new distribution: {new_score_new}')
                    model_new_score['on_new_data'] = float(new_score_new)
                    
                    # Save metrics with proper format
                    with open('slow_graph_metrics.yaml', 'w') as f:
                        yaml.dump({'model_new_score': model_new_score}, f)
                        
                except FileNotFoundError as e:
                    print(f"Data file not found: {str(e)}")
                    print("Please ensure all required data files are present in the dataset folder.")
                except Exception as e:
                    print(f"Error during model training/evaluation: {str(e)}")
            
            fixes_made:
            - "Fixed missing y_train argument in model.fit()"
            - "Added complete on_old_data metric to model_new_score"
            - "Implemented proper train/test data loading"
            - "Added robust error handling with specific exceptions"
            - "Ensured evaluation uses ONLY test sets (no training data evaluation)"
            - "Added random_state for reproducibility"
            - "Implemented proper retraining workflow with combined data"
            - "Added comprehensive data validation"
            
            validation_steps:
            - "Verify all required data files exist in dataset folder"
            - "Check that model.fit() receives both X and y arguments"
            - "Confirm evaluation uses only test sets, never training data"
            - "Validate metrics dictionary structure is complete"
            - "Ensure no cross-validation on test data"
            - "Check that metrics file is created with correct format"
            - "Verify error handling covers common failure cases"
            """
        },
        {
            "input": """
            current_code: |
                import pandas as pd
                import yaml
                from sklearn.model_selection import cross_val_score
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score
                
                # Load data
                dataset_folder = "datasets/financial"
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                model = RandomForestClassifier()
                
                # Use CV on test data - WRONG!
                score = cross_val_score(model, X_test_old, y_test_old, cv=5).mean()
                print(f'Score: {score}')
            
            error_output: |
                ValueError: cross_val_score requires fitted model or training data, invalid evaluation methodology
            """,
            "output": """
            fixed_code: |
                import pandas as pd
                import yaml
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score
                
                # Initialize metrics dictionary
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                try:
                    # Load ALL required data from specified folder
                    dataset_folder = "datasets/financial"
                    X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                    X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                    y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                    y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                    
                    # Load new data for retraining
                    X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                    y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                    X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                    y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                    
                    # Combine training data for retraining
                    X_train = pd.concat([X_train_old, X_train_new])
                    y_train = pd.concat([y_train_old, y_train_new])
                    
                    # Train model on combined training data
                    model_new = RandomForestClassifier(random_state=42)
                    model_new.fit(X_train, y_train)
                    
                    # FIXED: Evaluate on test sets ONLY, no cross-validation on test data
                    new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                    print(f'New model trained and evaluated on old distribution: {new_score_old}')
                    model_new_score['on_old_data'] = float(new_score_old)
                    
                    new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                    print(f'New model evaluated on new distribution: {new_score_new}')
                    model_new_score['on_new_data'] = float(new_score_new)
                    
                    # Save metrics with proper format
                    with open('slow_graph_metrics.yaml', 'w') as f:
                        yaml.dump({'model_new_score': model_new_score}, f)
                        
                except FileNotFoundError as e:
                    print(f"Required data file not found: {str(e)}")
                    print("Ensure all train/test files for old and new data exist.")
                except ValueError as e:
                    print(f"Data validation error: {str(e)}")
                    print("Check data shapes and formats match expected structure.")
                except Exception as e:
                    print(f"Unexpected error during model training: {str(e)}")
            
            fixes_made:
            - "CRITICAL: Removed invalid cross-validation on test data"
            - "Added proper training data loading for model fitting"
            - "Implemented correct train/test evaluation methodology"
            - "Added complete data loading for retraining workflow"
            - "Fixed evaluation to use only dedicated test sets"
            - "Added proper error handling for missing files"
            - "Implemented correct metrics structure and saving"
            - "Added data validation error handling"
            
            validation_steps:
            - "CRITICAL: Verify no cross-validation is used on test data"
            - "Confirm model is trained on training data, evaluated on test data"
            - "Check that evaluation methodology follows train/test separation"
            - "Validate all required data files are loaded"
            - "Ensure metrics follow proper structure"
            - "Verify no data leakage in preprocessing or evaluation"
            - "Test error handling for various failure scenarios"
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    system_prompt = """
    You are an expert ML engineer specializing in debugging and fixing ML code while ensuring fair evaluation practices.
    
    Context: You have code that produced an error and needs to be fixed while maintaining proper metrics tracking and evaluation methodology.
    
    CRITICAL EVALUATION REQUIREMENTS - FIXES MUST ENFORCE THESE:
    1. NEVER allow cross-validation on test data (X_test_old, X_test_new)
    2. NEVER allow evaluation on training data for final metrics
    3. NEVER allow fitting preprocessing separately on test data
    4. ONLY allow evaluation using: model.predict(X_test_old) and model.predict(X_test_new)
    5. Ensure proper train/test separation throughout the pipeline
    6. Fix any data leakage issues in the original code
    
    COMMON ERRORS TO FIX:
    1. **Methodology Errors** (HIGHEST PRIORITY):
       - cross_val_score(model, X_test, y_test) - Remove and replace with proper test evaluation
       - model.score(X_train, y_train) for final metrics - Replace with test set evaluation
       - GridSearchCV on test data - Replace with training data validation
       - Separate scaler.fit_transform on test data - Fix to use scaler.transform
    
    2. **Technical Errors**:
       - Missing function arguments (e.g., model.fit(X) without y)
       - Import errors and missing libraries
       - Variable scope and naming issues
       - File path and data loading problems
       - Incorrect metrics format or saving
    
    3. **Logic Errors**:
       - Missing data loading steps
       - Incorrect data concatenation or splitting
       - Wrong metrics dictionary structure
       - Missing error handling
    
    Critical Requirements:
    1. Maintain correct metrics format:
       model_new_score:
           on_new_data: [score]
           on_old_data: [score]
    2. Ensure robust error handling with specific exception types
    3. Include proper metrics file saving to 'slow_graph_metrics.yaml'
    4. Use clear, descriptive print messages for model evaluation
    5. Keep sklearn-compatible code structure
    6. MANDATORY: Fix any evaluation methodology violations
    
    Objective: Fix the code while:
    1. **PRIORITY 1**: Correcting any unfair evaluation practices
    2. Addressing specific ML library errors and conflicts
    3. Handling parameter incompatibilities and missing arguments
    4. Fixing training and evaluation setup issues
    5. Maintaining proper data processing flow with train/test separation
    6. Adding comprehensive error prevention and handling
    7. Preserving dataset folder paths and data loading patterns
    8. Implementing complete retraining workflow when needed
    
    Fix Strategy Guidelines:
    1. **Evaluation Fixes**: Replace any test data CV with proper test set evaluation
    2. **Error Prevention**: Add try-catch blocks for common failure modes
    3. **Data Validation**: Ensure all required data files are loaded
    4. **Complete Workflow**: Implement full train/test pipeline if missing
    5. **Robust Implementation**: Handle edge cases and provide informative errors
    6. **Code Quality**: Maintain clean, well-commented, production-ready code
    
    Response Format: Format your response as YAML with:
    fixed_code: |
      [COMPLETE FIXED CODE WITH FAIR EVALUATION]
    fixes_made:
      - [List of specific fixes, highlighting evaluation methodology fixes]
    validation_steps:
      - [Steps to validate fixes, emphasizing evaluation methodology validation]
    
    The fixed code must:
    1. Eliminate any unfair evaluation practices
    2. Handle the specific error that occurred
    3. Implement proper error handling and validation
    4. Follow strict train/test separation principles
    5. Use only test sets for final evaluation
    6. Provide clear, informative output messages
    
    REMEMBER: Any fixed code that still violates fair evaluation principles will be rejected. 
    The primary goal is to ensure methodologically sound ML code that follows proper evaluation practices.
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    system_prompt = textwrap.dedent(system_prompt).strip()
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    return final_prompt


def prompt_summarize_model_docs() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            A random forest classifier.

            A random forest is a meta estimator that fits a number of decision tree
            classifiers on various sub-samples of the dataset and uses averaging to
            improve the predictive accuracy and control over-fitting.
            
            Parameters
            ----------
            n_estimators : int, default=100
                The number of trees in the forest.
            
            criterion : {"gini", "entropy", "log_loss"}, default="gini"
                The function to measure the quality of a split. Supported criteria are
                "gini" for the Gini impurity and "log_loss" and "entropy" both for the
                Shannon information gain
            
            max_depth : int, default=None
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure or until all leaves contain less than
                min_samples_split samples.
                
            [... rest of documentation ...]
            """,
            "output": """
            model_params = RandomForestClassifier(
                n_estimators=200,              # Number of trees in forest. Try: 100, 200, 500
                criterion='entropy',           # Split quality metric: 'gini' (default), 'entropy', 'log_loss'
                max_depth=10,                  # Max tree depth. None for full depth, or 10, 20, 50
                min_samples_split=5,           # Min samples to split node. Try: 2 (default), 5, 10
                min_samples_leaf=3,            # Min samples at leaf. Try: 1 (default), 3, 5
                max_features='log2',           # Features per split: 'sqrt', 'log2', None, or int
                max_leaf_nodes=50,             # Max leaf nodes. None (default) or 50, 100, 500
                min_impurity_decrease=0.01,    # Min impurity decrease. Try: 0.0, 0.01, 0.05
                bootstrap=True,                # Bootstrap samples. True (default) or False
                oob_score=True,                # Out-of-bag scoring if bootstrap=True
                n_jobs=-1,                     # CPU cores to use. -1 for all cores
                random_state=42,               # Random seed for reproducibility
                class_weight='balanced',       # Class weights: None, 'balanced', 'balanced_subsample'
                ccp_alpha=0.01                 # Complexity parameter. Try: 0.0, 0.01, 0.05
            )
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    system_prompt = """
    You are an expert ML engineer tasked with summarizing model documentation into a concise parameter listing.

    Objective: Convert detailed model documentation into a compact parameter listing with:
    1. Common parameter values as constructor arguments
    2. Brief explanations as inline comments
    3. Sample values to try for each parameter
    4. Only the most important parameters

    Format: Create a model instantiation with:
    - One parameter per line
    - Inline comments explaining each parameter
    - 2-3 suggested values in comments
    - Aligned comments for readability

    Example Format:
    model = ModelName(
        param1=value,                  # Brief explanation. Try: val1, val2, val3
        param2=value,                  # Brief explanation. Try: val1, val2, val3
    )

    Keep comments concise but informative. Include default values and common alternatives.
    Only output the formatted code block, no additional text.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt


def prompt_fix_code_slow() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_code: |
                import pandas as pd
                import lightgbm as lgb
                from sklearn.metrics import accuracy_score
                import yaml
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                model_old = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6
                )
                
                model_old.fit(
                    X_train_old, 
                    y_train_old,
                    early_stopping_rounds=10,
                    eval_set=[(X_test_old, y_test_old)]
                )
            
            error_output: |
                TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'
            """,
            "output": """
            fixed_code: |
                import pandas as pd
                import lightgbm as lgb
                from sklearn.metrics import accuracy_score
                import yaml
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Initialize metrics dictionaries
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                # Configure LightGBM with appropriate parameters
                model_old = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                
                # Train using proper LightGBM API
                model_old.fit(
                    X_train_old, 
                    y_train_old,
                    eval_set=[(X_test_old, y_test_old)],
                    callbacks=[lgb.early_stopping(10)]  # Use callbacks for early stopping
                )
                
                # Evaluate model
                old_score_old = model_old.score(X_test_old, y_test_old)
                model_new_score['on_old_data'] = float(old_score_old)
                print(f'Model trained and evaluated on old distribution: {old_score_old}')
                
                # Save metrics
                with open('slow_graph_metrics.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            fixes_made:
              - Replaced early_stopping_rounds with proper callback
              - Added random_state for reproducibility
              - Maintained evaluation set functionality
              - Added metrics tracking and saving
              - Added clarifying comments
            
            validation_steps:
              - Verify early stopping is working
              - Check model convergence
              - Validate prediction functionality
              - Confirm metrics are saved correctly
            """
        },
        {
            "input": """
            current_code: |
                import pandas as pd
                import xgboost as xgb
                from sklearn.metrics import accuracy_score
                import yaml
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train = pd.read_csv(f"{dataset_folder}/X_train.csv")
                y_train = pd.read_csv(f"{dataset_folder}/y_train.csv").squeeze("columns")
                
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4
                )
                
                model.fit(X_train, y_train, eval_metric=['error'])
                y_pred = model.predict(X_test_old)
            
            error_output: |
                ValueError: Invalid eval_metric: must be list of tuples or dict
            """,
            "output": """
            fixed_code: |
                import pandas as pd
                import xgboost as xgb
                from sklearn.metrics import accuracy_score
                import yaml
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train = pd.read_csv(f"{dataset_folder}/X_train.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train = pd.read_csv(f"{dataset_folder}/y_train.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Initialize metrics dictionaries
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                # Configure XGBoost with proper parameters
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    eval_metric='error',  # Move to params
                    random_state=42
                )
                
                # Train with proper evaluation setup
                model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_train, y_train), (X_test_old, y_test_old)],
                    verbose=False
                )
                
                # Evaluate model
                score = model.score(X_test_old, y_test_old)
                model_new_score['on_old_data'] = float(score)
                print(f'Model trained and evaluated on old distribution: {score}')
                
                # Save metrics
                with open('slow_graph_metrics.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            fixes_made:
              - Moved eval_metric to constructor
              - Added proper eval_set format
              - Added random_state
              - Added verbose parameter
              - Added metrics tracking and saving
              - Added proper data loading
              - Improved code organization
            
            validation_steps:
              - Check evaluation metrics
              - Verify prediction shapes
              - Monitor training progress
              - Validate metrics file creation
            """
        },
        {
            "input": """
            current_code: |
                import pandas as pd
                from sklearn.ensemble import VotingClassifier
                import lightgbm as lgb
                import xgboost as xgb
                import yaml
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train = pd.read_csv(f"{dataset_folder}/X_train.csv")
                y_train = pd.read_csv(f"{dataset_folder}/y_train.csv").squeeze("columns")
                X_val = pd.read_csv(f"{dataset_folder}/X_val.csv")
                y_val = pd.read_csv(f"{dataset_folder}/y_val.csv").squeeze("columns")
                
                estimators = [
                    ('lgb', lgb.LGBMClassifier()),
                    ('xgb', xgb.XGBClassifier(eval_metric='error'))
                ]
                
                model = VotingClassifier(
                    estimators=estimators,
                    voting='soft'
                )
                
                model.fit(X_train, y_train, xgb__eval_set=[(X_val, y_val)])
            
            error_output: |
                ValueError: Unknown parameter for estimator lgb: eval_set
            """,
            "output": """
            fixed_code: |
                import pandas as pd
                from sklearn.ensemble import VotingClassifier
                import lightgbm as lgb
                import xgboost as xgb
                import yaml
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train = pd.read_csv(f"{dataset_folder}/X_train.csv")
                y_train = pd.read_csv(f"{dataset_folder}/y_train.csv").squeeze("columns")
                X_val = pd.read_csv(f"{dataset_folder}/X_val.csv")
                y_val = pd.read_csv(f"{dataset_folder}/y_val.csv").squeeze("columns")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Initialize metrics dictionary
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                # Configure base models separately
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                )
                
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    eval_metric='error',
                    random_state=42
                )
                
                # Create and train base models first
                lgb_model.fit(X_train, y_train)
                xgb_model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Create voting classifier with trained models
                estimators = [
                    ('lgb', lgb_model),
                    ('xgb', xgb_model)
                ]
                
                model = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    n_jobs=-1
                )
                
                # Final fit of ensemble
                model.fit(X_train, y_train)
                
                # Evaluate ensemble
                score = model.score(X_test_old, y_test_old)
                model_new_score['on_old_data'] = float(score)
                print(f'Ensemble trained and evaluated on old distribution: {score}')
                
                # Save metrics
                with open('slow_graph_metrics.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            fixes_made:
              - Separated base model training
              - Removed problematic parameter passing
              - Added proper eval_set handling
              - Added random states
              - Added n_jobs for parallel processing
              - Added metrics tracking and saving
              - Added complete data loading
            
            validation_steps:
              - Verify base models trained correctly
              - Check ensemble predictions
              - Validate voting mechanism
              - Confirm metrics are saved correctly
            """
        }
    ]

    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    system_prompt = """
    You are an expert ML engineer specializing in fixing machine learning code errors.

    Context: You have:
    1. ML training code that produced an error during execution
    2. Dataset folder path from current code
    3. Requirements for metrics tracking and saving
    4. Error output to be addressed

    Objective: Fix the code while:
    1. Addressing specific ML library errors and conflicts
    2. Handling parameter incompatibilities
    3. Fixing training and evaluation setup issues
    4. Maintaining proper data processing flow
    5. Adding relevant error prevention
    6. Preserving dataset folder paths and data loading
    7. Implementing proper metrics tracking
    8. Improving code organization and documentation

    Critical Requirements:
    1. Maintain data loading paths from original code
    2. Use correct metrics format:
    model_new_score:
        on_new_data: [score]
        on_old_data: [score]
    3. Save metrics to 'slow_graph_metrics.yaml'
    4. Include proper error handling
    5. Ensure reproducibility with random states

    Common Issues to Handle:
    1. LightGBM/XGBoost parameter conflicts
    2. Early stopping implementation
    3. Evaluation metric setup
    4. Model fitting parameter errors
    5. Ensemble model training issues
    6. Data validation and preprocessing errors
    7. Metrics tracking and saving
    8. Data loading consistency

    Style: Provide clean, well-documented ML code with:
    1. Clear parameter organization
    2. Proper evaluation setup
    3. Error prevention mechanisms
    4. Informative comments
    5. Consistent structure
    6. Complete data loading
    7. Proper metrics handling
    8. Clear print messages

    Response Format: Format your response as YAML with:

    fixed_code: |
    [COMPLETE FIXED CODE]
    fixes_made:
    - [List of specific fixes]
    validation_steps:
    - [Steps to validate the fixes]

    The fixed code must:
    1. Include all necessary imports
    2. Maintain dataset folder paths
    3. Load all required data files
    4. Initialize metrics properly
    5. Save metrics in correct format
    6. Include proper error handling
    7. Add informative print messages

    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    return final_prompt

def prompt_parse_strategy_output() -> ChatPromptTemplate:
    """Creates a prompt to parse potentially unstructured strategy analysis output."""
    
    system_prompt = """
    You are an expert at parsing and structuring ML strategy analysis output.
    
    Your task is to take a potentially unstructured analysis output and extract:
    1. The recommended strategy (model_selection, hyperparameter_tuning, or ensemble_method)
    2. Key reasoning points
    3. Performance gaps identified
    4. Next steps recommended
    
    Even if the input is messy or poorly structured, identify the core elements.
    If specific elements are missing, make reasonable inferences based on context.
    
    Response Format: Format your response as YAML with:
    
    recommended_strategy: [strategy_name]
    reasoning: [key points identified]
    performance_gaps: 
      - [list of gaps]
    next_steps:
      - [list of steps]
    
    Keep your response focused and concise.
    """
    
    human_template = """
    Please parse the following analysis output and extract the key elements:
    
    {input}
    """
    
    system_prompt = textwrap.dedent(system_prompt).strip()
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template),
    ])


def prompt_analyze_improvement_needs() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_performance: |
                Old model trained and evaluated on the old distribution: 0.913
                Old model evaluated on the new distribution: 0.717
                
                Training new model on combined data...
                New model trained and evaluated on old distribution: 0.907
                New model evaluated on new distribution: 0.800
            
            strategy_results:
                model_selection:
                    tried: false
                    models_tried: []
                    best_accuracy: 0.0
                hyperparameter_tuning:
                    tried: false
                    best_accuracy: 0.0
                ensemble_method:
                    tried: false
                    best_accuracy: 0.0
            
            improvement_history:
                - strategy_type: "baseline"
                  improvements:
                    old_distribution: -0.006
                    new_distribution: 0.083
                  final_outcome: "accept"
            """,
            "output": """
            recommended_strategy: "model_selection"
            priority: "high"
            confidence: "high"
            
            reasoning: |
                1. Significant distribution gap detected (19.6% initial, 10.7% remaining)
                2. Baseline retraining shows strong adaptation capability (+8.3% on new data)
                3. Model architecture likely not optimal for distribution shift handling
                4. No model alternatives have been explored yet
                5. GradientBoosting or SVM could provide better sequential learning
                6. Current RandomForest may lack sufficient regularization for drift
            
            performance_analysis:
                initial_gap: 0.196
                current_gap: 0.107
                gap_reduction: 0.089
                adaptation_potential: "high"
                convergence_risk: "low"
                
            strategy_effectiveness_prediction:
                model_selection:
                    expected_improvement: "medium-high"
                    rationale: "Architecture change can address distribution shift fundamentally"
                hyperparameter_tuning:
                    expected_improvement: "medium"
                    rationale: "Can optimize current model but limited by architecture constraints"
                ensemble_method:
                    expected_improvement: "high"
                    rationale: "Best for final optimization but requires good base models first"
            
            risk_assessment:
                - "Model selection has moderate risk of regression on old distribution"
                - "Current 10.7% gap suggests room for architectural improvements"
                - "Strong baseline adaptation indicates dataset is learnable"
                
            tried_strategies: []
            untried_strategies: ["model_selection", "hyperparameter_tuning", "ensemble_method"]
            
            next_steps:
                - "Try GradientBoostingClassifier with early stopping for better drift handling"
                - "If GradientBoosting fails, try SVC with RBF kernel for non-linear patterns"
                - "Consider ExtraTreesClassifier as alternative ensemble base learner"
                - "Reserve hyperparameter tuning for successful model architectures"
                - "Plan ensemble strategy combining best 2-3 models from selection phase"
                
            stopping_criteria:
                target_gap: 0.05
                max_strategies: 3
                min_improvement_threshold: 0.02
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    system_prompt = """
    You are an expert ML engineer and strategic advisor specializing in systematic model improvement analysis.
    
    Context: You have:
    1. Current model performance metrics on old and new distributions
    2. Results from previous improvement strategies with detailed history
    3. Strategy effectiveness tracking across attempts
    4. Available strategies to try with their historical performance
    
    Objective: Conduct sophisticated analysis to recommend the optimal next strategy based on:
    1. Performance pattern analysis and trend detection
    2. Strategy effectiveness prediction based on current model behavior
    3. Risk-reward assessment for each potential strategy
    4. Resource efficiency and expected ROI of strategies
    5. Stopping criteria and convergence detection
    
    Advanced Analysis Requirements:
    1. PERFORMANCE PATTERN ANALYSIS:
       - Calculate distribution gaps, adaptation rates, and convergence trends
       - Detect if model is overfitting, underfitting, or properly adapting
       - Analyze improvement velocity and diminishing returns patterns
       - Assess baseline retraining effectiveness as predictor of strategy success
    
    2. STRATEGY EFFECTIVENESS PREDICTION:
       - Predict expected improvement range for each untried strategy
       - Consider synergies between strategies (e.g., model selection + hyperparameter tuning)
       - Account for strategy ordering effects and dependency chains
       - Evaluate risk of regression on old distribution
    
    3. INTELLIGENT STRATEGY ORDERING:
       - Prioritize strategies based on current model's weaknesses
       - Consider computational cost vs expected benefit
       - Account for strategy failure rates and backup plans
       - Optimize for overall improvement trajectory, not just next step
    
    4. CONVERGENCE AND STOPPING ANALYSIS:
       - Detect when diminishing returns suggest stopping
       - Identify if current performance is near theoretical limits
       - Recommend stopping criteria based on gap analysis
       - Suggest strategy combination approaches for final optimization
    
    5. CONTEXTUAL INTELLIGENCE:
       - Analyze why previous strategies succeeded or failed
       - Detect model architecture limitations vs parameter optimization needs
       - Identify distribution shift characteristics (covariate, label, concept drift)
       - Recommend strategies aligned with specific drift patterns
    
    Available Strategies (with intelligent ordering):
    1. Model Selection: Architecture changes, algorithm replacement
    2. Hyperparameter Tuning: Parameter optimization within architecture
    3. Ensemble Methods: Multi-model combination strategies
    
    Strategy Selection Logic:
    - If large gap + no architectures tried → Model Selection (high priority)
    - If good architecture + suboptimal params → Hyperparameter Tuning
    - If multiple good models available → Ensemble Methods
    - If diminishing returns across strategies → Stop or advanced ensembles
    - If consistent failures → Re-evaluate problem formulation
    
    Response Format: Format your response as YAML with:
    recommended_strategy: [strategy_name]
    priority: ["high", "medium", "low"]
    confidence: ["high", "medium", "low"]
    
    reasoning: |
      [Detailed numbered analysis with specific insights]
    
    performance_analysis:
      initial_gap: [float]
      current_gap: [float]
      gap_reduction: [float]
      adaptation_potential: ["high", "medium", "low"]
      convergence_risk: ["high", "medium", "low"]
    
    strategy_effectiveness_prediction:
      model_selection:
        expected_improvement: [prediction]
        rationale: [reasoning]
      hyperparameter_tuning:
        expected_improvement: [prediction]
        rationale: [reasoning]
      ensemble_method:
        expected_improvement: [prediction]
        rationale: [reasoning]
    
    risk_assessment:
      - [List of potential risks and mitigation strategies]
    
    tried_strategies: [list]
    untried_strategies: [list]
    
    next_steps:
      - [Specific, actionable recommendations with model suggestions]
    
    stopping_criteria:
      target_gap: [float]
      max_strategies: [int]
      min_improvement_threshold: [float]
    
    The analysis must:
    1. Provide quantitative performance gap analysis
    2. Make data-driven strategy predictions
    3. Consider computational efficiency and ROI
    4. Account for strategy interaction effects
    5. Provide clear stopping criteria and success metrics
    6. Give specific model recommendations, not just generic advice
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    system_prompt = textwrap.dedent(system_prompt).strip()
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    return final_prompt


def prompt_model_selection_change() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_code: |
                from sklearn.ensemble import RandomForestClassifier
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                model.fit(X_train_old, y_train_old)
            
            execution_output: |
                Old model trained and evaluated on the old distribution: 0.913
                Old model evaluated on the new distribution: 0.717
                
                Training new model on combined data...
                New model trained and evaluated on old distribution: 0.907
                New model evaluated on new distribution: 0.800
            
            models_tried: []
            
            previous_metrics:
                model_old_score:  # Previous baseline performance
                    on_new_data: 0.717
                    on_old_data: 0.913
            """,
            "output": """
            model_name: "GradientBoostingClassifier"
            new_training_code: |
                import yaml
                import pandas as pd
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.metrics import accuracy_score
                
                # Initialize metrics dictionary
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Load new data
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                
                # Train new model on combined data
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                model_new = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=42
                )
                
                model_new.fit(X_train, y_train)
                
                # Evaluate new model on old test set
                new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                print(f'New model trained and evaluated on old distribution: {new_score_old}')
                model_new_score['on_old_data'] = float(new_score_old)
                
                # Evaluate new model on new test set
                new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score_new}')
                model_new_score['on_new_data'] = float(new_score_new)
                
                # Save metrics
                with open('slow_graph_metrics.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            changes_made:
              - "Switched from RandomForest to GradientBoosting"
              - "Added early stopping with n_iter_no_change"
              - "Implemented subsample=0.8 for better generalization"
              - "Updated metrics format to track performance"
              
            parameters:
                n_estimators: 100
                learning_rate: 0.1
                max_depth: 6
                subsample: 0.8
                validation_fraction: 0.1
                n_iter_no_change: 10
            
            rationale: |
                GradientBoostingClassifier selected because:
                1. Better sequential learning for handling distribution shifts
                2. Built-in early stopping capabilities
                3. Stochastic gradient boosting for improved generalization
                4. Strong performance on both old and new distributions
                5. Maintains sklearn API compatibility
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    system_prompt = """
    You are an expert ML engineer specializing in model selection and implementation in sklearn.
    
    Context: You have:
    1. Current model code and configuration
    2. Performance metrics from previous run
    3. List of previously tried models
    4. Dataset folder path from current code
    
    Objective: Select and implement a new model architecture that:
    1. Better handles distribution shifts
    2. Maintains performance on old distribution
    3. Improves performance on new distribution
    4. Uses best practices for the chosen model
    5. Preserves the dataset folder path from input code
    
    CRITICAL EVALUATION REQUIREMENTS - YOUR CODE WILL BE REJECTED IF THESE ARE VIOLATED:
    1. NEVER use cross-validation on test data (X_test_old, X_test_new)
    2. NEVER evaluate on training data for final metrics
    3. NEVER fit preprocessing (scalers, etc.) separately on test data
    4. NEVER use GridSearchCV or RandomizedSearchCV on test data
    5. ONLY evaluate using: model.predict(X_test_old) and model.predict(X_test_new)
    6. Model selection should be based on domain knowledge, not test set performance
    7. Train/validation splits should ONLY be applied to training data, never test data
    
    VALID EVALUATION PATTERN EXAMPLE:
    ```python
    # CORRECT: Evaluate on separate test sets
    new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
    new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
    
    # CORRECT: Use internal validation for model tuning
    model = GradientBoostingClassifier(validation_fraction=0.1, n_iter_no_change=10)
    
    # CORRECT: If using manual validation, split only training data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2)
    ```
    
    INVALID PATTERNS TO AVOID:
    ```python
    # WRONG: Model selection based on test set performance
    for model in models:
        score = model.score(X_test_old, y_test_old)  # Don't choose model this way
    
    # WRONG: CV on test data
    score = cross_val_score(model, X_test_old, y_test_old, cv=5)
    
    # WRONG: GridSearch on test data
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(X_test_old, y_test_old)
    
    # WRONG: Evaluation on training data
    score = accuracy_score(y_train, model.predict(X_train))
    
    # WRONG: Separate preprocessing on test data
    scaler.fit_transform(X_test_old)  # Should be scaler.transform(X_test_old)
    ```
    
    Critical Requirements:
    1. Only save new model metrics to 'slow_graph_metrics.yaml'
    2. Use metrics format:
       model_new_score:
           on_new_data: [score]
           on_old_data: [score]
    3. Previous baseline metrics are provided in input
    4. Only use sklearn-compatible models
    5. Maintain data loading paths from original code
    6. MANDATORY: Evaluate ONLY on X_test_old and X_test_new test sets
    7. Base model selection on theoretical advantages, not test set peeking
    
    Model Selection Guidelines:
    1. Choose models based on theoretical advantages for distribution shift
    2. Consider model's ability to handle distribution shifts (e.g., boosting, regularization)
    3. Use proper regularization techniques
    4. Implement early stopping when available (validation_fraction, n_iter_no_change)
    5. Set appropriate default parameters based on best practices
    6. Don't recompute old model metrics
    7. Keep dataset folder path consistent
    8. Avoid models already tried (check models_tried list)
    9. Select models with built-in validation mechanisms when possible
    10. Consider ensemble-friendly models if ensemble strategy might be next
    
    Recommended Models for Distribution Shift:
    - GradientBoostingClassifier: Sequential learning, early stopping
    - LogisticRegression: Regularization, interpretability
    - SVC: Kernel methods, regularization
    - ExtraTreesClassifier: Randomization, ensemble diversity
    - AdaBoostClassifier: Adaptive boosting
    - XGBClassifier: Advanced boosting (if available)
    
    Response Format: Format your response as YAML with:
    model_name: [selected_model_name]
    new_training_code: |
      [COMPLETE IMPLEMENTATION WITH FAIR EVALUATION]
    changes_made:
      - [List of significant changes]
    parameters:
      [Dictionary of key parameters]
    rationale: |
      [Detailed explanation of model choice based on theoretical advantages]
    
    The implementation must:
    1. Only compute and save new model metrics using proper test set evaluation
    2. Use provided baseline metrics for comparison
    3. Include clear parameter documentation
    4. Implement proper evaluation WITHOUT data leakage
    5. Follow train/test separation principles strictly
    6. Base model choice on theoretical merits, not test set performance
    
    REMEMBER: Any code that violates fair evaluation principles will be automatically rejected. 
    Only evaluate on dedicated test sets that were never used for training or validation.
    Model selection should be based on domain knowledge and theoretical advantages, not test set peeking.
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    system_prompt = textwrap.dedent(system_prompt).strip()
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    return final_prompt


def prompt_hyperparameter_tuning() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_code: |
                from sklearn.ensemble import GradientBoostingClassifier
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                model.fit(X_train, y_train)
            
            execution_output: |
                Old model trained and evaluated on the old distribution: 0.913
                Old model evaluated on the new distribution: 0.717
                
                Training new model on combined data...
                New model trained and evaluated on old distribution: 0.907
                New model evaluated on new distribution: 0.800
            
            current_params: {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6
            }
            
            previous_metrics:
                model_old_score:  # Previous baseline performance
                    on_new_data: 0.717
                    on_old_data: 0.913
            """,
            "output": """
            hyperparameters:
                n_estimators: 200
                learning_rate: 0.05
                max_depth: 4
                min_samples_split: 50
                subsample: 0.8
                validation_fraction: 0.1
                n_iter_no_change: 10
                tol: 0.01
            
            new_training_code: |
                import yaml
                import pandas as pd
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.metrics import accuracy_score
                
                # Initialize metrics dictionary
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Load new data
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                
                # Train new model on combined data
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                # Configure model with optimized hyperparameters
                model_new = GradientBoostingClassifier(
                    n_estimators=200,          # Increased for better convergence
                    learning_rate=0.05,        # Reduced for better generalization
                    max_depth=4,               # Reduced to prevent overfitting
                    min_samples_split=50,      # More conservative splits
                    subsample=0.8,             # Added stochastic sampling
                    validation_fraction=0.1,   # Added validation monitoring
                    n_iter_no_change=10,       # Added early stopping
                    tol=0.01,                 # Convergence tolerance
                    random_state=42
                )
                
                model_new.fit(X_train, y_train)
                
                # Evaluate new model on old test set
                new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                print(f'New model trained and evaluated on old distribution: {new_score_old}')
                model_new_score['on_old_data'] = float(new_score_old)
                
                # Evaluate new model on new test set
                new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score_new}')
                model_new_score['on_new_data'] = float(new_score_new)
                
                # Save new model metrics
                with open('slow_graph_metrics.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            changes_made:
              - "Increased n_estimators to 200 for better convergence"
              - "Reduced learning_rate to 0.05 for smoother learning"
              - "Reduced max_depth to 4 for better generalization"
              - "Added min_samples_split=50 for robust splits"
              - "Added subsample=0.8 for stochastic sampling"
              - "Implemented early stopping mechanism"
              
            rationale: |
                Parameter adjustments focus on:
                1. Better generalization with reduced tree depth and learning rate
                2. More robust training with conservative splits and subsampling
                3. Improved convergence monitoring with early stopping
                4. Better adaptation to distribution shifts through stochastic sampling
                5. Balance between model capacity and generalization
            """
        }
    ]
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    system_prompt = """
    You are an expert ML engineer specializing in hyperparameter optimization in sklearn.
    
    Context: You have:
    1. Current model code with parameters
    2. Previous baseline metrics from working memory
    3. Current parameter configuration
    4. Model performance history
    5. Dataset folder path from current code
    
    Objective: Optimize model hyperparameters to:
    1. Improve model robustness across distributions
    2. Better handle distribution shifts
    3. Maintain performance on old distribution
    4. Improve performance on new distribution
    5. Preserve the dataset folder path from input code
    
    CRITICAL EVALUATION REQUIREMENTS - YOUR CODE WILL BE REJECTED IF THESE ARE VIOLATED:
    1. NEVER use cross-validation on test data (X_test_old, X_test_new)
    2. NEVER evaluate on training data for final metrics
    3. NEVER fit preprocessing (scalers, etc.) separately on test data
    4. NEVER use GridSearchCV or RandomizedSearchCV on test data
    5. ONLY evaluate using: model.predict(X_test_old) and model.predict(X_test_new)
    6. Hyperparameter tuning should use validation_fraction or internal CV, NOT test data
    7. Train/validation splits should ONLY be applied to training data, never test data
    
    VALID EVALUATION PATTERN EXAMPLE:
    ```python
    # CORRECT: Evaluate on separate test sets
    new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
    new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
    
    # CORRECT: Use internal validation for tuning
    model = GradientBoostingClassifier(validation_fraction=0.1, n_iter_no_change=10)
    
    # CORRECT: If using manual validation, split only training data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2)
    ```
    
    INVALID PATTERNS TO AVOID:
    ```python
    # WRONG: GridSearch on test data
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(X_test_old, y_test_old)
    
    # WRONG: CV on test data
    score = cross_val_score(model, X_test_old, y_test_old, cv=5)
    
    # WRONG: Manual validation using test data
    train_test_split(X_test_old, y_test_old, test_size=0.2)
    
    # WRONG: Evaluation on training data
    score = accuracy_score(y_train, model.predict(X_train))
    
    #  WRONG: Separate preprocessing on test data
    scaler.fit_transform(X_test_old)  # Should be scaler.transform(X_test_old)
    ```
    
    Critical Requirements:
    1. Only save new model metrics to 'slow_graph_metrics.yaml'
    2. Use metrics format:
       model_new_score:
           on_new_data: [score]
           on_old_data: [score]
    3. Previous baseline metrics are provided in input
    4. Only tune parameters available in sklearn models
    5. Maintain data loading paths from original code
    6. MANDATORY: Evaluate ONLY on X_test_old and X_test_new test sets
    7. Use model's internal validation mechanisms when available
    
    Parameter Tuning Guidelines:
    1. Balance model capacity with generalization
    2. Consider regularization parameters
    3. Implement early stopping when available (validation_fraction, n_iter_no_change)
    4. Use validation monitoring through model parameters, not external CV
    5. Consider stochastic variants of parameters
    6. Don't recompute old model metrics
    7. Keep dataset folder path consistent
    8. If manual hyperparameter search is needed, only use training data for validation
    9. Prefer models with built-in validation over external grid search
    
    Response Format: Format your response as YAML with:
    hyperparameters:
      [parameter_name]: [value]
    new_training_code: |
      [COMPLETE IMPLEMENTATION WITH FAIR EVALUATION]
    changes_made:
      - [List of parameter changes]
    rationale: |
      [Detailed explanation of parameter choices]
    
    The implementation must:
    1. Only compute and save new model metrics using proper test set evaluation
    2. Use provided baseline metrics for comparison
    3. Include clear parameter documentation
    4. Implement proper evaluation WITHOUT data leakage
    5. Follow train/test separation principles strictly
    6. Use model's internal validation mechanisms when possible
    
    REMEMBER: Any code that violates fair evaluation principles will be automatically rejected. 
    Only evaluate on dedicated test sets that were never used for training or validation.
    Hyperparameter tuning should use internal model validation or manual splits of training data only.
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    system_prompt = textwrap.dedent(system_prompt).strip()
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    return final_prompt


def prompt_ensemble_method() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_code: |
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                from sklearn.ensemble import GradientBoostingClassifier
                
                model = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=42
                )
                model.fit(X_train, y_train)
            
            execution_output: |
                New model trained and evaluated on old distribution: 0.907
                New model evaluated on new distribution: 0.800
            
            strategy_results:
                model_selection:
                    tried: true
                    models_tried: ["RandomForest", "GradientBoosting"]
                hyperparameter_tuning:
                    tried: true
                    best_params:
                        n_estimators: 200
                        learning_rate: 0.05
                        max_depth: 4
                        
            previous_metrics:
                model_old_score:  # Previous baseline performance
                    on_new_data: 0.717
                    on_old_data: 0.913
            """,
            "output": """
            ensemble_type: "stacking"
            estimators:
            - name: "gradient_boost"
                class: "GradientBoostingClassifier"
                params:
                    n_estimators: 200
                    learning_rate: 0.05
                    max_depth: 4
            - name: "random_forest"
                class: "RandomForestClassifier"
                params:
                    n_estimators: 200
                    max_depth: 8
                    class_weight: "balanced"
                    
            new_training_code: |
                import yaml
                import pandas as pd
                from sklearn.ensemble import (
                    RandomForestClassifier, 
                    GradientBoostingClassifier, 
                    StackingClassifier
                )
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score
                
                # Initialize metrics dictionary
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Load new data
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                
                # Train new model on combined data
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                # Define base estimators
                estimators = [
                    ('gb', GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42
                    )),
                    ('rf', RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        class_weight='balanced',
                        random_state=42
                    ))
                ]
                
                # Create stacking ensemble
                model_new = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(class_weight='balanced'),
                    stack_method='predict_proba',
                    cv=5
                )
                
                # Train the ensemble
                model_new.fit(X_train, y_train)
                
                # Evaluate new model on old test set
                new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                print(f'New model trained and evaluated on old distribution: {new_score_old}')
                model_new_score['on_old_data'] = float(new_score_old)
                
                # Evaluate new model on new test set
                new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score_new}')
                model_new_score['on_new_data'] = float(new_score_new)
                
                # Save new model metrics
                with open('slow_graph_metrics.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            changes_made:
              - "Implemented StackingClassifier with GradientBoosting and RandomForest"
              - "Added LogisticRegression meta-learner"
              - "Used predict_proba for probabilistic stacking"
              - "Added balanced class weights"
              - "Implemented 5-fold cross-validation for ensemble training"
              
            rationale: |
                Ensemble strategy:
                1. Combines successful models from previous attempts
                2. Stacking leverages strengths of different models
                3. Cross-validation reduces overfitting risk
                4. Probabilistic stacking for better uncertainty handling
                5. Balanced weights for better distribution handling
                6. Logistic meta-learner for interpretable combination
            """
        }
    ]
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    system_prompt = """
    You are an expert ML engineer specializing in ensemble methods and model combinations.
    
    Context: You have:
    1. Current model code and configuration
    2. Previous models and parameters tried
    3. Previous baseline metrics from working memory
    4. Strategy results from previous attempts
    5. Dataset folder path from current code
    
    Objective: Create an ensemble that:
    1. Combines successful previous approaches
    2. Improves robustness across distributions
    3. Maintains performance on old distribution
    4. Enhances performance on new distribution
    5. Preserves the dataset folder path from input code
    
    CRITICAL EVALUATION REQUIREMENTS - YOUR CODE WILL BE REJECTED IF THESE ARE VIOLATED:
    1. NEVER use cross-validation on test data (X_test_old, X_test_new)
    2. NEVER evaluate on training data for final metrics
    3. NEVER fit preprocessing (scalers, etc.) separately on test data
    4. ONLY evaluate using: model.predict(X_test_old) and model.predict(X_test_new)
    5. Cross-validation is ONLY allowed during ensemble training (e.g., cv parameter in StackingClassifier)
    6. Train/validation splits should ONLY be applied to training data, never test data
    
    VALID EVALUATION PATTERN EXAMPLE:
    ```python
    # CORRECT: Evaluate on separate test sets
    new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
    new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
    
    # CORRECT: CV during ensemble training
    model = StackingClassifier(estimators=estimators, cv=5)
    ```
    
    INVALID PATTERNS TO AVOID:
    ```python
    #  WRONG: CV on test data
    score = cross_val_score(model, X_test_old, y_test_old, cv=5)
    
    #  WRONG: Evaluation on training data
    score = accuracy_score(y_train, model.predict(X_train))
    
    #  WRONG: Separate preprocessing on test data
    scaler.fit_transform(X_test_old)  # Should be scaler.transform(X_test_old)
    ```
    
    Critical Requirements:
    1. Only save new model metrics to 'slow_graph_metrics.yaml'
    2. Use metrics format:
       model_new_score:
           on_new_data: [score]
           on_old_data: [score]
    3. Previous baseline metrics are provided in input
    4. Only use sklearn ensemble methods
    5. Maintain data loading paths from original code
    6. MANDATORY: Evaluate ONLY on X_test_old and X_test_new test sets
    
    Ensemble Guidelines:
    1. Choose appropriate ensemble type (stacking, voting). Import required classes (VotingClassifier, StackingClassifier)
    2. Consider diversity in base estimators
    3. Use cross-validation ONLY for ensemble training (cv parameter), NEVER for evaluation
    4. Implement proper weighting strategies
    5. Consider probabilistic combinations
    6. Don't recompute old model metrics
    7. Keep dataset folder path consistent
    8. Do not use BaggingClassifier
    9. Ensure all preprocessing is fitted on training data only
    
    Response Format: Format your response as YAML with:
    ensemble_type: [type_name]
    estimators:
      - name: [estimator_name]
        class: [estimator_class]
        params: [parameters]
    new_training_code: |
      [COMPLETE IMPLEMENTATION WITH FAIR EVALUATION]
    changes_made:
      - [List of ensemble design choices]
    rationale: |
      [Detailed explanation of ensemble strategy]
    
    The implementation must:
    1. Only compute and save new model metrics using proper test set evaluation
    2. Use provided baseline metrics for comparison
    3. Include clear estimator configuration
    4. Implement proper ensemble evaluation WITHOUT data leakage
    5. Follow train/test separation principles strictly
    
    REMEMBER: Any code that violates fair evaluation principles will be automatically rejected. 
    Only evaluate on dedicated test sets that were never used for training or validation.
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    system_prompt = textwrap.dedent(system_prompt).strip()
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    return final_prompt

