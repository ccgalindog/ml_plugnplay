import pandas as pd
from typing import Any, Dict, List, Tuple
from src.pipelines.training_pipelines import StandardTrainerPipeline


def cross_validate(df: pd.DataFrame,
                   list_models: List,
                   list_model_names: List,
                   target: str,
                   metrics_config: Dict,
                   num_folds: int=5,
                   artifacts: Dict={},
                   numerical_features: List=None,
                   categorical_features: List=None,
                   preprocessing_graph: List=None,
                   test_size: float=0.2
                  ) -> Tuple[pd.DataFrame, Dict]:
  """
  Perform cross-validation on a given list of models.
  Args:
    list_models: List - List of models to be evaluated.
    list_model_names: List - List of names for the models.
    scaler: Any - Scaler object from sklearn.
    create_features: bool - Flag to decide if new features should be created
      or just use the raw features provided initially.
    num_folds: int - Number of folds for cross-validation.
    selected_features: List - Features to use in training, if None, all features
      are used.
  Returns:
    metrics_df: pd.DataFrame - DataFrame containing the evaluation metrics for
      each model and fold.
    all_models: Dict - Dictionary containing the trained models for each fold.
  """
  # TODO: Make load artifacts general
  metrics_df = pd.DataFrame()
  all_models = {}

  for k_fold in range(num_folds):
    # For each fold, create a new train-validation split
    # Then fit the preprocessor steps to the train part and apply to val part
    print(f'Fold {k_fold}')

    training_pipeline = StandardTrainerPipeline(df.copy(),
                                                numerical_features,
                                                categorical_features,
                                                target,
                                                preprocessing_graph,
                                                artifacts.copy(),
                                                list_models.copy(),
                                                list_model_names,
                                                metrics_config,
                                                test_size
                                               )
    training_pipeline.run()

    fold_metrics = training_pipeline.metrics_df
    fitted_models = training_pipeline.fitted_models

    fold_metrics['k_fold'] = k_fold
    metrics_df = pd.concat([metrics_df, fold_metrics], axis=0)
    all_models[k_fold] = fitted_models

  return metrics_df, all_models
