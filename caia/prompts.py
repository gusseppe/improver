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


def prompt_generate_retraining_code() -> ChatPromptTemplate:
    examples = [
        {
            "input": '''
            reference_training_code: |
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier

                # load the reference data
                dataset_folder = "datasets/financial"
                X_train_reference = pd.read_csv(f"{dataset_folder}/X_train_reference.csv")
                X_test_reference = pd.read_csv(f"{dataset_folder}/X_test_reference.csv")
                y_train_reference = pd.read_csv(f"{dataset_folder}/y_train_reference.csv").squeeze("columns")
                y_test_reference = pd.read_csv(f"{dataset_folder}/y_test_reference.csv").squeeze("columns")

                model_reference = RandomForestClassifier(random_state=42)

                model_reference.fit(X_train_reference, y_train_reference)

                # Test the model on the initial test set
                initial_accuracy = model_reference.score(X_test_reference, y_test_reference)

                print(f'Model trained and evaluated on the reference distribution: {initial_accuracy}')

            new_data: |
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
            ''',
            "output": '''
            new_training_code: |
                import yaml
                from sklearn.ensemble import RandomForestClassifier

                metrics = {
                    'model_reference': {},
                    'model_new': {},
                    'difference_score_averages': {}
                }

                # load the reference data
                X_train_reference = pd.read_csv(f"{dataset_folder}/X_train_reference.csv")
                X_test_reference = pd.read_csv(f"{dataset_folder}/X_test_reference.csv")
                y_train_reference = pd.read_csv(f"{dataset_folder}/y_train_reference.csv").squeeze("columns")
                y_test_reference = pd.read_csv(f"{dataset_folder}/y_test_reference.csv").squeeze("columns")

                # Train and evaluate reference model
                model_reference = RandomForestClassifier(random_state=42)
                model_reference.fit(X_train_reference, y_train_reference)

                # Test reference model on reference test set
                ref_score_reference = model_reference.score(X_test_reference, y_test_reference)
                print(f'Model trained and evaluated on the reference distribution: {ref_score_reference}')
                metrics['model_reference']['score_reference_data'] = float(ref_score_reference)

                # Test reference model on new test set
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")

                ref_score_new = model_reference.score(X_test_new, y_test_new)
                print(f'Reference model evaluated on the new distribution: {ref_score_new}')
                metrics['model_reference']['score_new_data'] = float(ref_score_new)

                # Calculate average score for reference model
                ref_score_average = (ref_score_reference + ref_score_new) / 2
                print(f'Average score of reference model: {ref_score_average}')
                metrics['model_reference']['score_average'] = float(ref_score_average)

                print("\\nTraining new model on combined data...")

                # load the new training data
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")

                # Combine datasets
                X_train = pd.concat([X_train_reference, X_train_new])
                y_train = pd.concat([y_train_reference, y_train_new])

                # Train new model on combined dataset
                model_new = RandomForestClassifier(random_state=42)
                model_new.fit(X_train, y_train)

                # Test new model on reference test set
                new_score_reference = model_new.score(X_test_reference, y_test_reference)
                print(f'New model evaluated on reference distribution: {new_score_reference}')
                metrics['model_new']['score_reference_data'] = float(new_score_reference)

                # Test new model on new test set
                new_score_new = model_new.score(X_test_new, y_test_new)
                print(f'New model evaluated on new distribution: {new_score_new}')
                metrics['model_new']['score_new_data'] = float(new_score_new)

                # Calculate average score for new model
                new_score_average = (new_score_reference + new_score_new) / 2
                print(f'Average score of new model: {new_score_average}')
                metrics['model_new']['score_average'] = float(new_score_average)

                # Calculate score difference
                score_difference = new_score_average - ref_score_average
                print(f'\\nScore difference: {score_difference}')
                metrics['difference_score_averages']['score_average'] = float(score_difference)
                
                # Save metrics to yaml file
                with open('retraining_metrics.yaml', 'w') as f:
                    yaml.dump(metrics, f)
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

    Context: Given a reference training code and new data loading code, generate a new training code that retrains the model.

    Objective: Create a new training code (new_training_code) that:
    1. First evaluates the reference model on both distributions:
       - Trains on reference data
       - Tests on reference test set
       - Tests on new (drifted) test set
       - Calculates average performance
    2. Then trains a new model on combined data and evaluates it:
       - Uses the same ML model and parameters
       - Trains on combined dataset (reference + new data)
       - Tests on reference test set
       - Tests on new (drifted) test set
       - Calculates average performance
    3. Prints performance metrics and data shapes at each step
    4. Saves the performance metrics to a YAML file named 'retraining_metrics.yaml' using the following structure:
       model_reference:
         score_reference_data: [score]
         score_new_data: [score]
         score_average: [score]
       model_new:
         score_reference_data: [score]
         score_new_data: [score]
         score_average: [score]
       difference_score_averages:
         score_average: [score]
    
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
            criticality_analysis: |
                arguments_for_high_criticality:
                  - Significant drift in Income, Loan Term, Interest Rate features
                  - SHAP values changes in key features
                  - Interest Rate distribution changes significant
                arguments_for_low_criticality:
                  - Age and Credit Score features stable
                  - Loan Amount distribution unchanged
                criticality_score:
                  score: 4
                  reason: "High-concern issues requiring prompt attention"

            execution_output: |
                Model trained and evaluated on the reference distribution: 0.913
                Reference model evaluated on the new distribution: 0.550
                Average score of reference model: 0.732
                
                Training new model on combined data...
                New model evaluated on reference distribution: 0.907
                New model evaluated on new distribution: 0.533
                Average score of new model: 0.720
                
                Score difference: -0.012

            model_code: |
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

            improvement_history: |
                - change: "Basic RandomForest implementation"
                  outcome: "needs improvement"
                  accuracy_delta: -0.012
            """,
            "output": """
            insights:
              performance_gaps:
                - High performance on reference data (0.913) but poor on new data (0.550)
                - Model showing significant generalization issues
                - Performance degradation of 36.3% on new distribution
                - Combined training not helping generalization (-0.012 delta)
              
              model_limitations:
                - Default RandomForest parameters not robust enough
                - Basic ensemble size (n_estimators=100 by default)
                - Unlimited tree depth may cause overfitting
                - Default min_samples_split may be too aggressive
                - Missing control for tree randomness
              
              hyperparameter_recommendations:
                primary_changes:
                  - n_estimators: 500  # Increase ensemble size for robustness
                  - max_depth: 15      # Control tree depth to prevent overfitting
                  - min_samples_split: 10  # More conservative splits
                  - min_samples_leaf: 4    # Ensure leaf node stability
                  - max_features: 'sqrt'   # Increase tree randomization
                  - bootstrap: True        # Enable bootstrapping
                  - class_weight: 'balanced'  # Handle any class imbalance
                
                alternative_parameters:
                  - n_estimators: 1000  # Very large ensemble
                  - max_depth: 20       # Deeper trees if needed
                  - min_samples_split: 20  # Very conservative splits
                
              alternative_models:
                gradient_boosting:
                  rationale: "Better handling of complex patterns and drift"
                  suggested_config:
                    - model: "XGBClassifier"
                    - n_estimators: 300
                    - learning_rate: 0.01
                    - max_depth: 8
                    - subsample: 0.8
                    
                lightgbm:
                  rationale: "Faster training with similar performance"
                  suggested_config:
                    - model: "LGBMClassifier"
                    - n_estimators: 400
                    - learning_rate: 0.01
                    - num_leaves: 128
                    - subsample: 0.8
                
              improvement_priority:
                1: "Increase RandomForest robustness with primary parameter changes"
                2: "Test alternative RandomForest parameters if primary fails"
                3: "Evaluate XGBoost as alternative model"
                4: "Consider LightGBM if training time becomes issue"
              
              expected_impacts:
                - Increased ensemble size should improve stability
                - Controlled tree depth should reduce overfitting
                - Conservative splits should improve generalization
                - Alternative models might handle drift better
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
    You are an expert ML engineer analyzing model performance and monitoring data.

    Context: You have access to:
    1. Semantic memory (model code, documentation, dataset info)
    2. Episodic memory (recent model behavior)
    3. Monitoring summary (drift and performance analysis)

    Objective: Provide comprehensive insights about:
    1. Performance gaps and their causes
    2. Model limitations and weaknesses
    3. Potential improvements based on monitoring
    4. Specific recommendations for changes
    5. Expected impact of changes

    Key Analysis Points:
    1. Performance gaps between reference and new data
    2. Feature drift impacts
    3. Model parameter limitations
    4. Improvement recommendations
    5. Expected outcomes

    Style: Provide precise, quantitative recommendations with exact parameter values.

    Response Format: Format your response as YAML with:

    insights:
      performance_gaps:
        - [Detailed performance analysis]
      
      drift_impact:
        - [Analysis of feature drift effects]
      
      model_limitations:
        - [Current limitations]
      
      improvement_recommendations:
        architecture_changes:
          - [Model architecture recommendations]
        parameter_adjustments:
          - [Parameter change suggestions]
        
      expected_impacts:
        - [Expected outcomes]

    Focus on connecting monitoring insights to specific improvement recommendations.
    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
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
                        'model_reference': {},
                        'model_new': {},
                        'difference_score_averages': {}
                    }
                    
                    # Train reference model
                    model_reference = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
                    model_reference.fit(X_train_reference, y_train_reference)
                    
            execution_results:
                output: |
                    Model trained and evaluated on the reference distribution: 0.902
                    Reference model evaluated on the new distribution: 0.617
                    Average score of reference model: 0.759
                    Training new model on combined data...
                    New model evaluated on reference distribution: 0.902
                    New model evaluated on new distribution: 0.642
                    Average score of new model: 0.772
                    Score difference: 0.013
                reference_score: 0.902
                new_distribution_score: 0.617
                score_difference: 0.013
                success: true
                
            distilled_insights: |
                insights:
                    performance_gaps:
                        - Large gap between reference and new distribution
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
                    'model_reference': {},
                    'model_new': {},
                    'difference_score_averages': {}
                }
                
                # Load reference data
                dataset_folder = "datasets/financial"
                X_train_reference = pd.read_csv(f"{dataset_folder}/X_train_reference.csv")
                X_test_reference = pd.read_csv(f"{dataset_folder}/X_test_reference.csv")
                y_train_reference = pd.read_csv(f"{dataset_folder}/y_train_reference.csv").squeeze("columns")
                y_test_reference = pd.read_csv(f"{dataset_folder}/y_test_reference.csv").squeeze("columns")
                
                # Configure and train reference model with enhanced robustness
                model_reference = RandomForestClassifier(
                    n_estimators=500,           # Increased for stability
                    max_depth=15,               # Prevent overfitting
                    min_samples_split=20,       # Conservative splits
                    min_samples_leaf=10,        # Stable leaves
                    max_features='sqrt',        # Feature randomization
                    class_weight='balanced',    # Handle imbalance
                    bootstrap=True,             # Enable bootstrapping
                    random_state=42
                )
                model_reference.fit(X_train_reference, y_train_reference)
                
                # Evaluate reference model
                y_pred_reference = model_reference.predict(X_test_reference)
                ref_score_reference = accuracy_score(y_test_reference, y_pred_reference)
                print(f'Model trained and evaluated on the reference distribution: {ref_score_reference}')
                metrics['model_reference']['score_reference_data'] = float(ref_score_reference)
                
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                y_pred_new = model_reference.predict(X_test_new)
                ref_score_new = accuracy_score(y_test_new, y_pred_new)
                print(f'Reference model evaluated on the new distribution: {ref_score_new}')
                metrics['model_reference']['score_new_data'] = float(ref_score_new)
                
                ref_score_average = (ref_score_reference + ref_score_new) / 2
                print(f'Average score of reference model: {ref_score_average}')
                metrics['model_reference']['score_average'] = float(ref_score_average)
                
                print("\\nTraining new model on combined data...")
                
                # Load and combine new data
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                X_train = pd.concat([X_train_reference, X_train_new])
                y_train = pd.concat([y_train_reference, y_train_new])
                
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
                y_pred_reference_new = model_new.predict(X_test_reference)
                new_score_reference = accuracy_score(y_test_reference, y_pred_reference_new)
                print(f'New model evaluated on reference distribution: {new_score_reference}')
                metrics['model_new']['score_reference_data'] = float(new_score_reference)
                
                y_pred_new_new = model_new.predict(X_test_new)
                new_score_new = accuracy_score(y_test_new, y_pred_new_new)
                print(f'New model evaluated on new distribution: {new_score_new}')
                metrics['model_new']['score_new_data'] = float(new_score_new)
                
                new_score_average = (new_score_reference + new_score_new) / 2
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
                2. Tree depth control (15) to prevent overfitting to reference data
                3. Conservative split and leaf parameters to build more stable trees
                4. Feature randomization and class balancing to handle distribution changes
                These changes aim to create a more robust model that should generalize 
                better across distributions while maintaining good performance on the
                reference dataset.
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
                from sklearn.ensemble import GradientBoostingClassifier
                
                model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train_reference, y_train_reference)
                
            execution_output: |
                Model trained and evaluated on the reference distribution: 0.913
                Reference model evaluated on the new distribution: 0.525
                Average score of reference model: 0.719
                New model evaluated on new distribution: 0.533
                
            previous_metrics:
                accuracy: 0.85
            """,
            "output": """
            evaluation:
              metrics:
                accuracy_reference_distribution: 0.913
                accuracy_new_distribution: 0.525
                accuracy_average_reference: 0.719
                accuracy_new_model: 0.533
                accuracy_delta: 0.008
              analysis:
                - "High performance on reference data (0.913) but poor generalization to new data (0.525)"
                - "Small improvement on new distribution (+0.008)"
                - "Model struggles with distribution shift"
                - "Current hyperparameters may be too aggressive"
              risk_assessment:
                - "Significant performance gap between distributions"
                - "Current model architecture might not handle drift well"
                - "Risk of overfitting to reference distribution"
              recommendation:
                action: "reject"
                confidence: "high"
                reasoning: "Despite small improvement, performance gap is too large. Need to try different approach."
              next_steps:
                - "Try hyperparameter_tuning with more conservative settings"
                - "Consider ensemble_method if tuning fails"
                - "Evaluate model_selection with more robust architectures"
            """
        },
        {
            "input": """
            current_code: |
                from sklearn.ensemble import RandomForestClassifier
                
                model = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=15,
                    min_samples_split=10,
                    class_weight='balanced',
                    random_state=42
                )
                model.fit(X_train_combined, y_train_combined)
                
            execution_output: |
                Model trained and evaluated on the reference distribution: 0.895
                Reference model evaluated on the new distribution: 0.612
                Average score of reference model: 0.753
                New model evaluated on new distribution: 0.635
                
            previous_metrics:
                accuracy: 0.525
            """,
            "output": """
            evaluation:
              metrics:
                accuracy_reference_distribution: 0.895
                accuracy_new_distribution: 0.612
                accuracy_average_reference: 0.753
                accuracy_new_model: 0.635
                accuracy_delta: 0.110
              analysis:
                - "Significant improvement on new distribution (+0.110)"
                - "Better balance between distributions"
                - "Conservative parameters showing promise"
                - "Still room for improvement"
              risk_assessment:
                - "Performance gap still exists but reduced"
                - "May need fine-tuning of current approach"
                - "Could benefit from ensemble methods"
              recommendation:
                action: "accept"
                confidence: "medium"
                reasoning: "Notable improvement achieved, but can be enhanced further with available strategies"
              next_steps:
                - "Try ensemble_method to combine with previous model"
                - "Fine-tune hyperparameter_tuning if ensemble fails"
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
    You are an expert ML engineer evaluating model changes with THREE available improvement strategies:
    1. model_selection: Trying different model architectures
    2. hyperparameter_tuning: Optimizing model parameters
    3. ensemble_method: Combining multiple models

    Context: You're evaluating changes to improve model performance on drift data.
    You cannot collect more data or implement monitoring - focus ONLY on the three available strategies.

    Objective: Provide a comprehensive evaluation that:
    1. Analyzes performance metrics in detail
    2. Assesses the model's ability to handle distribution drift
    3. Makes clear recommendations from available strategies
    4. Suggests concrete next steps using ONLY the three strategies
    
    Evaluation Focus:
    - Reference vs new distribution performance
    - Improvement in handling drift
    - Model robustness and generalization
    - Strategy effectiveness

    Style: Provide quantitative analysis focused on distribution shift and improvement potential.

    Response Format: Format your response as YAML with:

    evaluation:
      metrics:
        accuracy_reference_distribution: [number]
        accuracy_new_distribution: [number]
        accuracy_average_reference: [number]
        accuracy_new_model: [number]
        accuracy_delta: [number]
      analysis:
        - [Key findings about distribution handling]
      risk_assessment:
        - [Potential risks with current approach]
      recommendation:
        action: ["accept" or "reject"]
        confidence: ["high", "medium", or "low"]
        reasoning: [Why accept/reject focusing on drift handling]
      next_steps:
        - [ONLY suggest steps from: model_selection, hyperparameter_tuning, ensemble_method]

    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    return final_prompt

# def prompt_evaluate_change() -> ChatPromptTemplate:
#     examples = [
#         {
#             "input": """
#             current_code: |
#                 from sklearn.ensemble import RandomForestClassifier
#                 from sklearn.preprocessing import StandardScaler
#                 from sklearn.feature_selection import SelectFromModel
                
#                 model = RandomForestClassifier(
#                     n_estimators=150,
#                     max_depth=10,
#                     min_samples_split=5,
#                     random_state=42
#                 )
#                 X_train_scaled = scaler.fit_transform(X_train)
#                 X_test_scaled = scaler.transform(X_test)
#                 selector = SelectFromModel(estimator=model, threshold='mean')
#                 X_train_selected = selector.fit_transform(X_train_scaled, y_train)
#                 X_test_selected = selector.transform(X_test_scaled)
#                 model.fit(X_train_selected, y_train)
                
#             execution_output: |
#                 Training completed successfully.
#                 Previous Model Accuracy: 0.82
#                 New Model Accuracy: 0.85
#                 Feature Count Before Selection: 20
#                 Feature Count After Selection: 12
#                 Training Time: 45.2 seconds
#                 Memory Usage: 128MB
            
#             previous_metrics:
#                 accuracy: 0.82
#                 training_time: 38.5
#             """,
#             "output": """
#             evaluation:
#               metrics:
#                 accuracy_previous: 0.82
#                 accuracy_new: 0.85
#                 accuracy_delta: 0.03
#                 feature_reduction: 0.40
#                 training_time_change: 0.17
#               analysis:
#                 - "Significant accuracy improvement (+3%)"
#                 - "Successful feature reduction (40% fewer features)"
#                 - "Acceptable increase in training time (17%)"
#                 - "Memory usage within reasonable bounds"
#               risk_assessment:
#                 - "Feature selection may impact model interpretability"
#                 - "Increased training time needs monitoring"
#                 - "Reduced feature set could affect edge cases"
#               recommendation:
#                 action: "accept"
#                 confidence: "high"
#                 reasoning: "Strong accuracy improvement with manageable tradeoffs"
#               next_steps:
#                 - "Monitor performance on edge cases"
#                 - "Consider feature importance analysis"
#                 - "Document selected features for domain experts"
#             """
#         },
#         {
#             "input": """
#             current_code: |
#                 from sklearn.ensemble import RandomForestClassifier
#                 from sklearn.preprocessing import StandardScaler
                
#                 model = RandomForestClassifier(
#                     n_estimators=300,
#                     max_depth=None,
#                     min_samples_split=2,
#                     random_state=42
#                 )
#                 X_train_scaled = scaler.fit_transform(X_train)
#                 X_test_scaled = scaler.transform(X_test)
#                 model.fit(X_train_scaled, y_train)
                
#             execution_output: |
#                 Training completed successfully.
#                 Previous Model Accuracy: 0.85
#                 New Model Accuracy: 0.852
#                 Training Time: 89.5 seconds
#                 Memory Usage: 256MB
                
#             previous_metrics:
#                 accuracy: 0.85
#                 training_time: 45.2
#             """,
#             "output": """
#             evaluation:
#               metrics:
#                 accuracy_previous: 0.85
#                 accuracy_new: 0.852
#                 accuracy_delta: 0.002
#                 training_time_change: 0.98
#               analysis:
#                 - "Minimal accuracy improvement (0.2%)"
#                 - "Training time nearly doubled"
#                 - "Significant increase in memory usage"
#                 - "Diminishing returns from increased complexity"
#               risk_assessment:
#                 - "Resource usage becoming concerning"
#                 - "Cost-benefit ratio unfavorable"
#                 - "Potential overfitting risk"
#               recommendation:
#                 action: "reject"
#                 confidence: "high"
#                 reasoning: "Marginal gains don't justify increased resource usage"
#               next_steps:
#                 - "Revert to previous configuration"
#                 - "Explore alternative optimization strategies"
#                 - "Consider different architecture changes"
#             """
#         }
#     ]

#     for example in examples:
#         example["input"] = textwrap.dedent(example["input"]).strip()
#         example["output"] = textwrap.dedent(example["output"]).strip()

#     example_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("human", "{input}"),
#             ("ai", "{output}"),
#         ]
#     )
    
#     few_shot_prompt = FewShotChatMessagePromptTemplate(
#         example_prompt=example_prompt,
#         examples=examples,
#     )

#     system_prompt = """
#     You are an expert machine learning engineer tasked with evaluating model changes.

#     Context: You have the modified model code, execution results, and previous performance metrics.

#     Objective: Provide a comprehensive evaluation of the changes that:
#     1. Analyzes performance metrics in detail
#     2. Assesses resource usage implications
#     3. Identifies potential risks and tradeoffs
#     4. Makes a clear recommendation with confidence level
#     5. Suggests next steps based on the outcome
    
#     The evaluation should consider:
#     - Accuracy improvements
#     - Training time changes
#     - Memory usage
#     - Code complexity
#     - Resource efficiency
#     - Potential risks
#     - Future implications

#     Style: Provide a detailed, data-driven analysis with clear metrics and reasoning.

#     Tone: Technical and objective, focusing on quantitative measures and practical implications.

#     Audience: ML engineers and technical stakeholders who need to make decisions about model changes.

#     Response Format: Format your response as YAML with the following structure:

#     evaluation:
#       metrics:
#         accuracy_previous: [number]
#         accuracy_new: [number]
#         accuracy_delta: [number]
#         [other relevant metrics]
#       analysis:
#         - [Key findings about performance]
#       risk_assessment:
#         - [Potential risks and concerns]
#       recommendation:
#         action: ["accept" or "reject"]
#         confidence: ["high", "medium", or "low"]
#         reasoning: [Detailed explanation]
#       next_steps:
#         - [Recommended actions]

#     Only provide the YAML-formatted output. Do not include any other explanation or commentary.
#     """

#     system_prompt = textwrap.dedent(system_prompt).strip()

#     final_prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         few_shot_prompt,
#         ("human", "{input}"),
#     ])

#     return final_prompt

def prompt_fix_code() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            current_code: |
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = RandomForestClassifier(n_estimators=150)
                model.fit(X_train_scaled)
            
            error_output: |
                TypeError: fit() missing 1 required positional argument: 'y'
            """,
            "output": """
            fixed_code: |
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Initialize and train model with both X and y
                model = RandomForestClassifier(n_estimators=150)
                model.fit(X_train_scaled, y_train)  # Added missing y_train argument
            
            fixes_made:
              - Added missing y_train argument to model.fit()
              - Added clarifying comments
              - Maintained original functionality
            
            validation_steps:
              - Verify y_train is available in scope
              - Check data shapes match
              - Ensure scaling is applied consistently
            """
        },
        {
            "input": """
            current_code: |
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                from sklearn.feature_selection import SelectFromModel
                
                scaler = StandardScaler()
                selector = SelectFromModel(RandomForestClassifier())
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_train_selected = selector.fit_transform(X_train_scaled)
                X_test_selected = selector.transform(X_test_scaled)
            
            error_output: |
                NameError: name 'X_test_scaled' is not defined
            """,
            "output": """
            fixed_code: |
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                from sklearn.feature_selection import SelectFromModel
                
                # Scale both training and test data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)  # Added missing scaling step
                
                # Feature selection with proper ordering
                base_model = RandomForestClassifier(random_state=42)
                selector = SelectFromModel(base_model)
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)  # Added y_train
                X_test_selected = selector.transform(X_test_scaled)
            
            fixes_made:
              - Added missing test data scaling step
              - Added y_train to selector.fit_transform()
              - Added random_state for reproducibility
              - Improved code organization
            
            validation_steps:
              - Verify all data transformations are in correct order
              - Check consistency between training and test preprocessing
              - Validate feature selection output shapes
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
    You are an expert machine learning engineer tasked with fixing code errors.

    Context: You have code that produced an error and the error message.

    Objective: Fix the code while:
    1. Addressing the specific error
    2. Maintaining original functionality
    3. Adding proper error prevention
    4. Improving code organization
    5. Adding relevant comments
    6. Suggesting validation steps

    Style: Provide clean, well-documented code with clear organization.

    Tone: Technical and precise, focusing on best practices and error prevention.

    Audience: ML engineers who need to understand and maintain the fixed code.

    Response Format: Format your response as YAML with the following structure:

    fixed_code: |
      [COMPLETE FIXED CODE]
    fixes_made:
      - [List of specific fixes applied]
    validation_steps:
      - [Steps to validate the fixes]

    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
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
                import lightgbm as lgb
                from sklearn.metrics import accuracy_score
                
                model_reference = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6
                )
                
                model_reference.fit(
                    X_train_reference, 
                    y_train_reference,
                    early_stopping_rounds=10,
                    eval_set=[(X_test_reference, y_test_reference)]
                )
            
            error_output: |
                TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'
            """,
            "output": """
            fixed_code: |
                import lightgbm as lgb
                from sklearn.metrics import accuracy_score
                
                # Configure LightGBM with appropriate parameters
                model_reference = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                
                # Train using proper LightGBM API
                model_reference.fit(
                    X_train_reference, 
                    y_train_reference,
                    eval_set=[(X_test_reference, y_test_reference)],
                    callbacks=[lgb.early_stopping(10)]  # Use callbacks for early stopping
                )
            
            fixes_made:
              - Replaced early_stopping_rounds with proper callback
              - Added random_state for reproducibility
              - Maintained evaluation set functionality
              - Added clarifying comments
            
            validation_steps:
              - Verify early stopping is working
              - Check model convergence
              - Validate prediction functionality
            """
        },
        {
            "input": """
            current_code: |
                import xgboost as xgb
                from sklearn.metrics import accuracy_score
                
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4
                )
                
                model.fit(X_train, y_train, eval_metric=['error'])
                y_pred = model.predict(X_test_reference)
            
            error_output: |
                ValueError: Invalid eval_metric: must be list of tuples or dict
            """,
            "output": """
            fixed_code: |
                import xgboost as xgb
                from sklearn.metrics import accuracy_score
                
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
                    eval_set=[(X_train, y_train), (X_test_reference, y_test_reference)],
                    verbose=False
                )
                
                # Make predictions
                y_pred = model.predict(X_test_reference)
            
            fixes_made:
              - Moved eval_metric to constructor
              - Added proper eval_set format
              - Added random_state
              - Added verbose parameter
              - Improved code organization
            
            validation_steps:
              - Check evaluation metrics
              - Verify prediction shapes
              - Monitor training progress
            """
        },
        {
            "input": """
            current_code: |
                from sklearn.ensemble import VotingClassifier
                import lightgbm as lgb
                import xgboost as xgb
                
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
                from sklearn.ensemble import VotingClassifier
                import lightgbm as lgb
                import xgboost as xgb
                
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
            
            fixes_made:
              - Separated base model training
              - Removed problematic parameter passing
              - Added proper eval_set handling
              - Added random states
              - Added n_jobs for parallel processing
            
            validation_steps:
              - Verify base models trained correctly
              - Check ensemble predictions
              - Validate voting mechanism
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

    Context: You have ML training code that produced an error during execution.

    Objective: Fix the code while:
    1. Addressing specific ML library errors and conflicts
    2. Handling parameter incompatibilities
    3. Fixing training and evaluation setup issues
    4. Maintaining proper data processing flow
    5. Adding relevant error prevention
    6. Improving code organization and documentation

    Common Issues to Handle:
    1. LightGBM/XGBoost parameter conflicts
    2. Early stopping implementation
    3. Evaluation metric setup
    4. Model fitting parameter errors
    5. Ensemble model training issues
    6. Data validation and preprocessing errors

    Style: Provide clean, well-documented ML code with:
    1. Clear parameter organization
    2. Proper evaluation setup
    3. Error prevention mechanisms
    4. Informative comments
    5. Consistent structure

    Response Format: Format your response as YAML with:

    fixed_code: |
      [COMPLETE FIXED CODE]
    fixes_made:
      - [List of specific fixes]
    validation_steps:
      - [Steps to validate the fixes]

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
            current_performance:
              Model trained and evaluated on the reference distribution: 0.85
              Reference model evaluated on the new distribution: 0.72
              Average score of reference model: 0.785
              
              Training new model on combined data...
              New model evaluated on reference distribution: 0.84
              New model evaluated on new distribution: 0.73
              Average score of new model: 0.785
              
              Score difference: 0.0
            
            improvement_history:
              - iteration: 0
                strategy: model_selection
                changes: "Switched to XGBoost"
                accuracy_change: 0.01
                status: reject
            
            strategy_results:
              model_selection:
                tried: true
                models_tried: ["XGBoost"]
                best_accuracy: 0.785
              hyperparameter_tuning:
                tried: false
                best_accuracy: 0.0
              ensemble_method:
                tried: false
                best_accuracy: 0.0
            
            threshold: 0.05
            
            monitoring_report:
              drift_analysis:
                - feature: "income"
                  drift_score: 0.15
                - feature: "credit_score" 
                  drift_score: 0.08
                - feature: "age"
                  drift_score: 0.03
            """,
            "output": """
            recommended_strategy: "hyperparameter_tuning"
            reasoning: |
                1. Model selection strategy has been tried with XGBoost but showed minimal improvement (0.01)
                2. Significant drift in income feature (0.15) suggests need for model robustness
                3. Hyperparameter tuning not yet attempted and could help adapt to drift
                4. Current performance shows good base accuracy but poor generalization
            performance_gaps:
              - "13% drop in accuracy on new distribution (0.85 -> 0.72)"
              - "No improvement from combined training (0.0 difference)"
              - "Current best accuracy (0.785) below target threshold (0.05 improvement needed)"
            tried_strategies: ["model_selection"]
            next_steps:
              - "Try hyperparameter tuning focusing on tree depth and sample weights"
              - "If unsuccessful, move to ensemble methods"
              - "Consider feature-specific model tuning for high-drift features"
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
    You are an expert ML engineer specializing in model improvement strategy analysis.

    Context: You have:
    1. Current model performance metrics
    2. History of improvement attempts
    3. Results from previous strategies
    4. Performance threshold target
    5. Monitoring report with drift analysis

    Objective: Analyze the current state and recommend the next best strategy to improve model performance.
    Consider:
    1. Performance gaps and patterns
    2. Previously tried strategies and their results
    3. Drift patterns in features
    4. Distance from target threshold
    5. Potential for each untried strategy

    Strategies (in typical order of application):
    1. Model Selection: Trying different algorithms
    2. Hyperparameter Tuning: Optimizing model configuration
    3. Ensemble Methods: Combining multiple models

    Style: Provide clear, data-driven recommendations with specific reasoning.

    Response Format: Format your response as YAML with:

    recommended_strategy: [strategy_name]
    reasoning: |
      [Detailed reasoning for recommendation]
    performance_gaps:
      - [List of specific gaps identified]
    tried_strategies:
      - [List of previously attempted strategies]
    next_steps:
      - [Specific next actions to take]

    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
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
                
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                model.fit(X_train, y_train)
            
            execution_output: |
                Model trained and evaluated on the reference distribution: 0.82
                Reference model evaluated on the new distribution: 0.71
                Average score of reference model: 0.765
            
            models_tried: []
            
            dataset_representation:
                num_features: 15
                num_samples: 50000
                feature_types: 
                    numerical: 10
                    categorical: 5
                target_type: "binary"
                missing_values: false
            """,
            "output": """
            model_name: "XGBoostClassifier"
            new_training_code: |
                import xgboost as xgb
                from sklearn.metrics import accuracy_score
                import yaml
                
                metrics = {
                    'model_reference': {},
                    'model_new': {},
                    'difference_score_averages': {}
                }
                
                # Configure XGBoost for balanced performance and generalization
                model_reference = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    random_state=42
                )
                
                # Train and evaluate reference model
                model_reference.fit(
                    X_train_reference, 
                    y_train_reference,
                    eval_set=[(X_test_reference, y_test_reference)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                
                # Test reference model
                ref_score_reference = accuracy_score(y_test_reference, model_reference.predict(X_test_reference))
                print(f'Model trained and evaluated on the reference distribution: {ref_score_reference}')
                metrics['model_reference']['score_reference_data'] = float(ref_score_reference)
                
                ref_score_new = accuracy_score(y_test_new, model_reference.predict(X_test_new))
                print(f'Reference model evaluated on the new distribution: {ref_score_new}')
                metrics['model_reference']['score_new_data'] = float(ref_score_new)
                
                ref_score_average = (ref_score_reference + ref_score_new) / 2
                print(f'Average score of reference model: {ref_score_average}')
                metrics['model_reference']['score_average'] = float(ref_score_average)
                
                print("\nTraining new model on combined data...")
                
                # Train new model on combined data
                X_train = pd.concat([X_train_reference, X_train_new])
                y_train = pd.concat([y_train_reference, y_train_new])
                
                model_new = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    random_state=42
                )
                
                model_new.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_test_reference, y_test_reference)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                
                # Evaluate new model
                new_score_reference = accuracy_score(y_test_reference, model_new.predict(X_test_reference))
                print(f'New model evaluated on reference distribution: {new_score_reference}')
                metrics['model_new']['score_reference_data'] = float(new_score_reference)
                
                new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score_new}')
                metrics['model_new']['score_new_data'] = float(new_score_new)
                
                new_score_average = (new_score_reference + new_score_new) / 2
                print(f'Average score of new model: {new_score_average}')
                metrics['model_new']['score_average'] = float(new_score_average)
                
                score_difference = new_score_average - ref_score_average
                print(f'\nScore difference: {score_difference}')
                metrics['difference_score_averages']['score_average'] = float(score_difference)
                
                with open('retraining_metrics.yaml', 'w') as f:
                    yaml.dump(metrics, f)
            
            changes_made:
              - "Switched from RandomForest to XGBoost"
              - "Added early stopping with validation set"
              - "Implemented balanced feature sampling"
              - "Added performance tracking metrics"
            
            rationale: |
                XGBoost selected because:
                1. Better handling of imbalanced data through gradient boosting
                2. Regularization parameters for drift adaptation
                3. Feature subsampling for better generalization
                4. Early stopping to prevent overfitting
                5. Suitable for dataset size (50K samples) and feature count (15)
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
    You are an expert ML engineer specializing in model selection and implementation.

    Context: You have:
    1. Current model code and configuration
    2. Performance metrics on reference and new data
    3. Dataset characteristics
    4. List of previously tried models

    Objective: Select and implement a new model architecture that:
    1. Better handles the observed performance gaps
    2. Is appropriate for the dataset characteristics
    3. Maintains the same input/output interface
    4. Implements proper evaluation metrics
    5. Uses best practices for the chosen model

    Model Selection Guidelines:
    1. Consider dataset size and feature types
    2. Balance complexity with interpretability
    3. Consider training speed and resource requirements
    4. Implement appropriate default parameters
    5. Include proper evaluation setup

    Style: Provide clean, well-structured implementation with:
    1. Proper imports
    2. Clear parameter settings
    3. Comprehensive evaluation
    4. Performance metrics logging

    Response Format: Format your response as YAML with:

    model_name: [selected_model_name]
    new_training_code: |
      [COMPLETE IMPLEMENTATION]
    changes_made:
      - [List of significant changes]
    rationale: |
      [Detailed explanation of model choice]

    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
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
                import xgboost as xgb
                
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                model.fit(X_train, y_train)
            
            execution_output: |
                Model trained and evaluated on the reference distribution: 0.84
                Reference model evaluated on the new distribution: 0.76
                Average score of reference model: 0.80
            
            current_params:
                n_estimators: 100
                learning_rate: 0.1
                max_depth: 6
            
            dataset_representation:
                num_features: 15
                num_samples: 50000
                feature_types: 
                    numerical: 10
                    categorical: 5
                target_type: "binary"
                class_balance: "imbalanced"
            """,
            "output": """
            hyperparameters:
                n_estimators: 200
                learning_rate: 0.05
                max_depth: 4
                min_child_weight: 3
                subsample: 0.8
                colsample_bytree: 0.8
                scale_pos_weight: 2.5
            
            new_training_code: |
                import xgboost as xgb
                from sklearn.metrics import accuracy_score
                import yaml
                
                metrics = {
                    'model_reference': {},
                    'model_new': {},
                    'difference_score_averages': {}
                }
                
                # Configure XGBoost with optimized parameters for robustness
                model_reference = xgb.XGBClassifier(
                    n_estimators=200,          # Increased for better convergence
                    learning_rate=0.05,        # Reduced for finer convergence
                    max_depth=4,               # Reduced to prevent overfitting
                    min_child_weight=3,        # Added to ensure robust splits
                    subsample=0.8,             # Added row sampling for robustness
                    colsample_bytree=0.8,      # Added column sampling
                    scale_pos_weight=2.5,      # Added for class imbalance
                    random_state=42
                )
                
                # Train and evaluate reference model
                model_reference.fit(
                    X_train_reference, 
                    y_train_reference,
                    eval_set=[(X_test_reference, y_test_reference)],
                    early_stopping_rounds=20,
                    verbose=False
                )
                
                # [Rest of the evaluation code remains the same...]
                
            changes_made:
              - "Doubled n_estimators for better convergence"
              - "Halved learning rate for finer steps"
              - "Reduced max_depth to control overfitting"
              - "Added min_child_weight for robust splits"
              - "Implemented sampling parameters"
              - "Added class weight scaling"
            
            rationale: |
                Parameter adjustments target:
                1. Overfitting control through reduced depth and increased min_child_weight
                2. Better generalization through sampling parameters
                3. Class imbalance handling with scale_pos_weight
                4. Finer convergence with lower learning rate and more estimators
                5. Robustness through row and column sampling
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
    You are an expert ML engineer specializing in hyperparameter optimization.

    Context: You have:
    1. Current model code with parameters
    2. Performance metrics on reference and new data
    3. Dataset characteristics
    4. Previous parameter configurations tried

    Objective: Optimize model hyperparameters to:
    1. Improve model robustness across distributions
    2. Reduce overfitting
    3. Handle dataset characteristics (class imbalance, feature types)
    4. Maintain reasonable training time
    5. Implement proper evaluation metrics

    Hyperparameter Tuning Guidelines:
    1. Focus on parameters that affect model complexity
    2. Consider dataset size when setting parameters
    3. Address any class imbalance issues
    4. Implement cross-validation where appropriate
    5. Use early stopping when possible
    6. Balance underfitting vs overfitting

    Style: Provide clean, well-structured implementation with:
    1. Clear parameter settings and comments
    2. Proper evaluation setup
    3. Performance tracking
    4. Early stopping where applicable

    Response Format: Format your response as YAML with:

    hyperparameters:
      [parameter_name]: [value]
    new_training_code: |
      [COMPLETE IMPLEMENTATION]
    changes_made:
      - [List of parameter changes]
    rationale: |
      [Detailed explanation of parameter choices]

    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
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
                import xgboost as xgb
                
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=42
                )
                model.fit(X_train, y_train)
            
            execution_output: |
                Model trained and evaluated on the reference distribution: 0.86
                Reference model evaluated on the new distribution: 0.78
                Average score of reference model: 0.82
            
            strategy_results:
                model_selection:
                    tried: true
                    models_tried: ["RandomForest", "XGBoost"]
                    best_accuracy: 0.82
                hyperparameter_tuning:
                    tried: true
                    best_accuracy: 0.82
            
            dataset_representation:
                num_features: 15
                num_samples: 50000
                feature_types: 
                    numerical: 10
                    categorical: 5
                target_type: "binary"
                class_balance: "imbalanced"
            """,
            "output": """
            ensemble_type: "stacking"
            new_training_code: |
                import xgboost as xgb
                from sklearn.ensemble import RandomForestClassifier, VotingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score
                import yaml
                
                metrics = {
                    'model_reference': {},
                    'model_new': {},
                    'difference_score_averages': {}
                }
                
                # Define base models with different strengths
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42
                )
                
                rf_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    random_state=42
                )
                
                # Create stacking ensemble
                estimators = [
                    ('xgb', xgb_model),
                    ('rf', rf_model)
                ]
                
                model_reference = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=[0.6, 0.4]  # Weight towards XGBoost which performed better
                )
                
                # Train and evaluate reference model
                model_reference.fit(X_train_reference, y_train_reference)
                
                # Test reference model
                ref_score_reference = accuracy_score(
                    y_test_reference, 
                    model_reference.predict(X_test_reference)
                )
                print(f'Model trained and evaluated on the reference distribution: {ref_score_reference}')
                metrics['model_reference']['score_reference_data'] = float(ref_score_reference)
                
                ref_score_new = accuracy_score(
                    y_test_new, 
                    model_reference.predict(X_test_new)
                )
                print(f'Reference model evaluated on the new distribution: {ref_score_new}')
                metrics['model_reference']['score_new_data'] = float(ref_score_new)
                
                ref_score_average = (ref_score_reference + ref_score_new) / 2
                print(f'Average score of reference model: {ref_score_average}')
                metrics['model_reference']['score_average'] = float(ref_score_average)
                
                print("\nTraining new model on combined data...")
                
                # Train new model on combined data
                X_train = pd.concat([X_train_reference, X_train_new])
                y_train = pd.concat([y_train_reference, y_train_new])
                
                model_new = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=[0.6, 0.4]
                )
                
                model_new.fit(X_train, y_train)
                
                # Evaluate new model
                new_score_reference = accuracy_score(
                    y_test_reference, 
                    model_new.predict(X_test_reference)
                )
                print(f'New model evaluated on reference distribution: {new_score_reference}')
                metrics['model_new']['score_reference_data'] = float(new_score_reference)
                
                new_score_new = accuracy_score(
                    y_test_new, 
                    model_new.predict(X_test_new)
                )
                print(f'New model evaluated on new distribution: {new_score_new}')
                metrics['model_new']['score_new_data'] = float(new_score_new)
                
                new_score_average = (new_score_reference + new_score_new) / 2
                print(f'Average score of new model: {new_score_average}')
                metrics['model_new']['score_average'] = float(new_score_average)
                
                score_difference = new_score_average - ref_score_average
                print(f'\nScore difference: {score_difference}')
                metrics['difference_score_averages']['score_average'] = float(score_difference)
                
                with open('retraining_metrics.yaml', 'w') as f:
                    yaml.dump(metrics, f)
            
            changes_made:
              - "Implemented soft voting ensemble"
              - "Combined XGBoost and RandomForest"
              - "Weighted models based on previous performance"
              - "Maintained same evaluation framework"
              - "Added comprehensive metrics tracking"
            
            rationale: |
                Ensemble design choices:
                1. Soft voting chosen for probability-based combination
                2. XGBoost and RF complement each other:
                   - XGBoost: Strong with numerical features
                   - RF: Good at handling categorical features
                3. Weights favor XGBoost (0.6) based on previous performance
                4. Both models configured for robustness
                5. Maintained evaluation framework for consistency
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
    You are an expert ML engineer specializing in ensemble methods and model combination strategies.

    Context: You have:
    1. Current model code and performance
    2. Results from previous improvement strategies
    3. Dataset characteristics
    4. Performance metrics across distributions

    Objective: Design and implement an ensemble approach that:
    1. Combines strengths of multiple models
    2. Improves generalization across distributions
    3. Handles dataset characteristics
    4. Maintains reasonable training time
    5. Provides robust predictions

    Ensemble Strategy Guidelines:
    1. Choose appropriate ensemble type:
       - Voting: Simple combination of predictions
       - Stacking: Meta-model learns optimal combination
       - Bagging: Parallel training on bootstrapped samples
       - Boosting: Sequential training focusing on errors
    2. Select complementary base models
    3. Configure appropriate weights/parameters
    4. Implement proper prediction aggregation
    5. Maintain evaluation framework

    Style: Provide clean, well-structured implementation with:
    1. Clear model definitions
    2. Proper ensemble setup
    3. Comprehensive evaluation
    4. Performance tracking

    Response Format: Format your response as YAML with:

    ensemble_type: [type_name]
    new_training_code: |
      [COMPLETE IMPLEMENTATION]
    changes_made:
      - [List of ensemble design choices]
    rationale: |
      [Detailed explanation of ensemble strategy]

    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    return final_prompt


def prompt_summarize_monitoring_report() -> ChatPromptTemplate:
    examples = [
        {
            "input": """
            drift_analysis:
              - feature: "income"
                drift_score: 0.15
                shapley_change: 0.03
                distribution_shift: "significant"
              - feature: "credit_score"
                drift_score: 0.08
                shapley_change: -0.02
                distribution_shift: "moderate"
              - feature: "age"
                drift_score: 0.03
                shapley_change: 0.01
                distribution_shift: "minimal"
              
            performance_metrics:
              accuracy_drop: 0.12
              feature_importance_changes: "high"
              data_quality_issues: "none"
            """,
            "output": """
            monitoring_summary:
              critical_drifts:
                - "Income shows significant drift (0.15) with increased importance (+0.03)"
                - "Credit score exhibits moderate drift (0.08) with decreased importance (-0.02)"
              
              stability_assessment:
                stable_features:
                  - "Age remains stable with minimal drift (0.03)"
                unstable_features:
                  - "Income (drift: 0.15, importance change: +0.03)"
                  - "Credit score (drift: 0.08, importance change: -0.02)"
              
              overall_impact:
                performance_degradation: "12% accuracy drop"
                importance_shifts: "Significant changes in feature importance"
                primary_concerns:
                  - "High drift in income feature"
                  - "Moderate drift in credit score"
                  - "Substantial accuracy degradation"
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
    You are an expert ML engineer analyzing model monitoring reports.

    Context: You have monitoring data that includes:
    1. Feature drift analysis
    2. Shapley value changes
    3. Distribution shifts
    4. Performance metrics

    Objective: Create a concise yet comprehensive summary that (smaller is better):
    1. Identifies critical drifts and changes
    2. Assesses feature stability
    3. Evaluates overall impact
    4. Highlights primary concerns

    Style: Provide quantitative analysis with clear priorities.

    Response Format: Format your response as YAML with:

    monitoring_summary:
      critical_drifts:
        - [Most important drift observations]
      
      stability_assessment:
        stable_features:
          - [Features with minimal changes]
        unstable_features:
          - [Features with significant changes]
      
      overall_impact:
        performance_degradation: [Summary of performance drop]
        importance_shifts: [Summary of importance changes]
        primary_concerns:
          - [Key issues to address]

    Only provide the YAML-formatted output. Do not include any other explanation or commentary.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    return final_prompt