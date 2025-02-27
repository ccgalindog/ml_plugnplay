from typing import Any, Tuple, List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from src.metrics.metrics import compute_metrics
from src.preprocessing.preprocessing import Preprocessor
from src.io.input import load_artifacts


def fit_model(model: Any,
              model_name: str,
              train_data: pd.DataFrame,
              validation_data: pd.DataFrame,
              target: str,
              dict_metric_functions: Dict) -> Tuple[Any, pd.DataFrame]:
  """
  Fit a given model to the training data and evaluate its performance on the
  validation data, then return the trained model and a DataFrame containing
  the evaluation metrics.
  Args:
    model: Any - the model to be fitted and evaluated.
    model_name: str - name of the model / algorithm being evaluated.
    train_data: pd.DataFrame - the training data.
    validation_data: pd.DataFrame - the validation data.
  Returns:
    model: Any - the trained model.
    metrics_df: pd.DataFrame - a DataFrame containing the computed metrics.

  """
  model.fit(train_data.drop(target, axis=1), train_data[target])
  y_train_pred = model.predict(train_data.drop(target, axis=1))
  y_val_pred = model.predict(validation_data.drop(target, axis=1))

  metrics_train = compute_metrics(model_name,
                                  'train',
                                  train_data[target],
                                  y_train_pred,
                                  dict_metric_functions)
  metrics_val = compute_metrics(model_name,
                                'validation',
                                validation_data[target],
                                y_val_pred,
                                dict_metric_functions)
  return model, pd.concat([metrics_train, metrics_val])


def cross_validate(df: pd.DataFrame,
                   list_models: List,
                   list_model_names: List,
                   scaler: Any,
                   create_features: bool,
                   target: str,
                   num_folds:int = 5,
                   selected_features=None) -> Tuple[pd.DataFrame, Dict]:
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
    
    df_train, df_val = train_test_split(df, test_size=0.2)

    train_pipeline = Preprocessor(df_train,
                                  'train',
                                  scaler=scaler,
                                  create_features=create_features)
    df_train_preproc = train_pipeline.preprocess()

    fitted_scaler = load_artifacts()

    validation_pipeline = Preprocessor(df_val,
                                      'val',
                                      scaler=fitted_scaler,
                                      create_features=create_features)
    df_val_preproc = validation_pipeline.preprocess()

    if selected_features is not None:
      df_train_preproc = df_train_preproc[selected_features + [target]]
      df_val_preproc = df_val_preproc[selected_features + [target]]

    fitted_models = {}
    for model, model_name in zip(list_models, list_model_names):
      # Train each model in the list, get its error metrics and add them to
      # the complete metrics dataset. The fitted model is also stored in
      # the all_models dict
      fitted_model, model_metrics = fit_model(model,
                                              model_name,
                                              df_train_preproc,
                                              df_val_preproc)
      model_metrics['k_fold'] = k_fold
      metrics_df = pd.concat([metrics_df, model_metrics], axis=0)
      fitted_models[model_name] = fitted_model

    all_models[k_fold] = fitted_models

  return metrics_df, all_models
