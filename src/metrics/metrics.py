import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error,\
                              mean_absolute_percentage_error, r2_score,\
                              root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
                            f1_score, roc_auc_score


def compute_metrics(model_name: str,
                    dataset_type: str,
                    y_real: np.array,
                    y_pred: np.array,
                    dict_metric_functions: Dict) -> pd.DataFrame:
  """
  Compute key regression metrics for a given model and dataset.
  Args:
    model_name: str - name of the model / algorithm being evaluated.
    dataset_type: str - type of dataset (train, validation, test).
    y_real: np.array - actual target values.
    y_pred: np.array - predicted target values.
  Returns:
    metrics_df: pd.DataFrame - a DataFrame containing the computed metrics.
  """
  metrics_dict = {'model' : [model_name], 'dataset' : [dataset_type]}
  for metric_name, function_metric in dict_metric_functions.items():
    metrics_dict[metric_name] = [function_metric(y_real, y_pred)]
  metrics_df = pd.DataFrame.from_dict(metrics_dict)
  return metrics_df


def default_regression_metrics():
  """
  Define the default regression metrics to be computed.
  Returns:
    dict_metric_functions: Dict - a dictionary containing the default regression
      metrics to be computed.
  """
  dict_metric_functions = {'mae' : mean_absolute_error,
                           'mse' : mean_squared_error,
                           'rmse' : root_mean_squared_error,
                           'mape' : mean_absolute_percentage_error,
                           'r2' : r2_score
                          }
  return dict_metric_functions

def default_classification_metrics():
  """
  Define the default classification metrics to be computed.
  Returns:
    dict_metric_functions: Dict - a dictionary containing the default
      classification metrics to be computed.
  """
  dict_metric_functions = {'accuracy' : accuracy_score,
                           'precision' : precision_score,
                           'recall' : recall_score,
                           'f1' : f1_score,
                           'roc_auc' : roc_auc_score
                          }
  return dict_metric_functions

def summarize_cv_metrics(metrics_df: pd.DataFrame,
                         sort_metric: str) -> pd.DataFrame:
  """
  Summarize the cross-validation metrics by dataset and model, by taking the
  average and standard deviation of the metric. 
  Args:
    metrics_df: pd.DataFrame - DataFrame containing the evaluation metrics for
      each model and fold.
  Returns:
    pd.DataFrame - DataFrame containing the summarized evaluation metrics for
      each model and dataset, where the metrics are shown as {mean +/- std}.
  """
  avg_metrics_cv = metrics_df.drop(columns=['k_fold'])\
                              .groupby(['dataset', 'model'])\
                              .agg(['mean', 'std']).reset_index()
  metrics_names = metrics_df.drop(columns=['k_fold', 'dataset', 'model']
                                  ).columns

  for metric in metrics_names:
    avg_metrics_cv[metric + '_cv'] = avg_metrics_cv[metric]['mean']\
                                                        .round(4).astype('str')\
                                      + ' +/- '\
                                      + avg_metrics_cv[metric]['std']\
                                                        .round(4).astype('str')

  avg_metrics_cv[sort_metric + '_mean'] = avg_metrics_cv[sort_metric]['mean']\
                                                        .round(4)
  return avg_metrics_cv.sort_values(['dataset', sort_metric + '_mean'])\
                          [['dataset', 'model']
                           + [metric + '_cv' for metric in metrics_names]
                          ]
