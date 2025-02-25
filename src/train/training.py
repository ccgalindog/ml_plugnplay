from typing import Any, Tuple, List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from src.metrics.metrics import compute_metrics
from src.preprocessing.preprocessing import Preprocessor
from src.io.input import load_artifacts


def fit_model(model: Any,
              model_name: str,
              train_data: pd.DataFrame,
              validation_data: pd.DataFrame,
              target: str) -> Tuple[Any, pd.DataFrame]:
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
                                  y_train_pred)
  metrics_val = compute_metrics(model_name,
                                'validation',
                                validation_data[target],
                                y_val_pred)
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


def objective_score(search_space):
  """
  Objective function to optimize, it returns the value of the loss function
  for some specific model iteration on some hyperparameters combination.
  Args:
    search_space - Grid of hyperparameters where the search will be performed
  Returns:
    loss - Value of the loss function for the given hyperparameters
    status - Status object indicating that the iteration went smoothly
  """
  # TODO: Make objects come from args
  print(search_space)
  try:
    model = LGBMRegressor(n_estimators=search_space['n_estimators'],
                          class_weight=search_space['class_weight'],
                          max_depth=int(search_space['max_depth']),
                          num_leaves=int(search_space['num_leaves']),
                          colsample_bytree=search_space['colsample_bytree'],
                          boosting_type=search_space['boosting_type'],
                          reg_alpha=search_space['reg_alpha'],
                          reg_lambda=search_space['reg_lambda'],
                          learning_rate=search_space['learning_rate'],
                          verbosity=-1
                        )
    fitted_model, model_metrics = fit_model(model,
                                            model_name,
                                            df_train_preproc[top_k_features + ['target']],
                                            df_val_preproc[top_k_features + ['target']]
                                          )
    
    test_mse = float(model_metrics[model_metrics['dataset']=='validation']['mse'])
    test_mae = float(model_metrics[model_metrics['dataset']=='validation']['mae'])
    test_mape = float(model_metrics[model_metrics['dataset']=='validation']['mape'])
    print(f'mse: {test_mse} - mae: {test_mae} - mape: {test_mape}')
  except:
    test_mse = 10000000
  return {'loss': test_mse, 'status': STATUS_OK }
