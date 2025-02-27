from lightgbm import LGBMRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from src.train.training import fit_model


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
