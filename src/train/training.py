from typing import Any, Tuple, List, Dict
import pandas as pd
from src.metrics.metrics import compute_metrics
import logging
logger = logging.getLogger(__name__)


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
  logger.info(f'Fitting model {model_name}')
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
  logger.info(f'Model {model_name} fitted')
  logger.info(f'Model metrics: {metrics_val.iloc[0].to_dict()}')
  return model, pd.concat([metrics_train, metrics_val])

def train_multiple_models(df_train_preproc: pd.DataFrame,
                          df_val_preproc: pd.DataFrame,
                          list_models: List,
                          list_model_names: List,
                          target_column: str,
                          metrics_config: Dict) -> Tuple[pd.DataFrame, Dict]:
  """
  Train multiple models and evaluate them on the validation set.
  Args:
    df_train_preproc: pd.DataFrame - preprocessed training data.
    df_val_preproc: pd.DataFrame - preprocessed validation data.
    list_models: List - List of models to be evaluated.
    list_model_names: List - List of names for the models.
    target_column: str - Name of the target column.
    metrics_config: Dict - Dictionary containing the metrics to be computed.
  Returns:
    metrics_df: pd.DataFrame - DataFrame containing the evaluation metrics for
      each model.
    fitted_models: Dict - Dictionary containing the trained models.
  """
  logger.info('Training models')
  metrics_df = pd.DataFrame()
  fitted_models = {}
  for model, model_name in zip(list_models, list_model_names):
    fitted_model, model_metrics = fit_model(model,
                                            model_name,
                                            df_train_preproc,
                                            df_val_preproc,
                                            target_column,
                                            metrics_config
                                          )
    metrics_df = pd.concat([metrics_df, model_metrics], axis=0)
    fitted_models[model_name] = fitted_model
  logger.info('Completed model training')
  return metrics_df, fitted_models
