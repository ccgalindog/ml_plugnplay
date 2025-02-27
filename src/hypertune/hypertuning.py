import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from functools import partial
from src.train.training import fit_model
from typing import Any
import logging
logger = logging.getLogger(__name__)


def objective_score(search_space: dict,
                    loss_metric: str,
                    model2tune: Any,
                    model_name: str,
                    df_trainset: pd.DataFrame,
                    df_valset: pd.DataFrame,
                    target_column: str,
                    metrics_config: dict
                   ) -> dict:
  """
  Objective function to optimize, it returns the value of the loss function
  for some specific model iteration on some hyperparameters combination.
  Args:
    search_space - Grid of hyperparameters where the search will be performed
  Returns:
    loss - Value of the loss function for the given hyperparameters
    status - Status object indicating that the iteration went smoothly
  """
  model = model2tune(**search_space)
    
  fitted_model, model_metrics = fit_model(model,
                                          model_name,
                                          df_trainset,
                                          df_valset,
                                          target_column,
                                          metrics_config
                                        )
  val_metrics = model_metrics[model_metrics['dataset']=='validation']\
                                    .drop(columns=['dataset', 'model'])\
                                    .iloc[0].to_dict()
  print(val_metrics)
  logger.info('Model metrics: {}'.format(val_metrics))
  iter_score = model_metrics\
                      [model_metrics['dataset']=='validation'][loss_metric]
  return {'loss': iter_score, 'status': STATUS_OK}


def hyperparameter_tunning(search_space: dict,
                           max_iterations: int,
                           loss_metric: str,
                           model2tune: Any,
                           model_name: str,
                           df_train_preproc: pd.DataFrame,
                           df_val_preproc: pd.DataFrame,
                           target_column: str,
                           metrics_config: dict,
                           objective_function: Any
                          ) -> dict:
  """
  Function to perform hyperparameter tunning using the hyperopt library.
  Args:
    search_space - Grid of hyperparameters where the search will be performed
    loss_metric - Metric to optimize
    model2tune - Model to tune
    model_name - Name of the model
    df_train_preproc - Preprocessed training dataset
    df_val_preproc - Preprocessed validation dataset
    target_column - Target column
    metrics_config - Configuration of the metrics to evaluate the model
  Returns:
    best_hyperparams - Best hyperparameters found during the search
  """
  logger.info('Starting hyperparameter tunning')
  trials = Trials()
  best_hyperparams = fmin(fn=partial(objective_function,
                                     loss_metric=loss_metric,
                                     model2tune=model2tune,
                                     model_name=model_name,
                                     df_trainset=df_train_preproc,
                                     df_valset=df_val_preproc,
                                     target_column=target_column,
                                     metrics_config=metrics_config
                                    ),
                          space=search_space,
                          algo=tpe.suggest,
                          max_evals=max_iterations,
                          trials=trials
                        )
  logger.info('Best trial: {}'.format(trials.best_trial))
  logger.info('Completed hyperparameter tunning')
  return best_hyperparams, trials
