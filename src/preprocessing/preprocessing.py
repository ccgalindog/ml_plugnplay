import pandas as pd
from typing import Any
from src.io.output import export_files
from src.utils.math import get_default_math_functions
import logging
logger = logging.getLogger(__name__)


class Preprocessor:
  """
  A class for all the preprocessing steps to apply to data in order to get it
  ready for the usage of machine learning models.
  """
  def __init__(self,
               df: pd.DataFrame,
               execution_mode: str,
               numerical_features: list,
               categorical_features: list,
               target_column: str,
               graph_preprocess: list,
               artifacts: dict={}):
    """
    Preprocessor constructor.
    Args:
      df: pd.DataFrame - Pandas DataFrame with the data to be preprocessed.
      execution_mode: str - Flag that defines the type of execution, either
        'train', 'validation' or 'test'.
      numerical_features: list - List with the names of the numerical features
        in the DataFrame.
      categorical_features: list - List with the names of the categorical
        features in the DataFrame.
      target_column: str - Name of the target column in the DataFrame.
      graph_preprocess: list - List with the names of the preprocessing steps
        to apply to the data. The possible values are:
          - 'scale_num': Scale the numerical features.
          - 'oh_encode_cat': Apply one-hot encoding to the categorical features.
          - 'individual_num': Create new features by applying non-linear
            functions to the numerical features.
          - 'combinated_num': Create new features by applying mathematical
            operations over each pair of numerical features.
      artifacts: dict - Dictionary to store the artifacts generated during the
        preprocessing. Some possible artifacts are:
          - 'scaler': The Sklearn scaler object.
          - 'oh_encoder': The Sklearn one-hot encoder object.
          - 'features_selected': The list of features selected for training. If
            not provided, all the features will be used.
    """
    self.df = df.copy()
    self.execution_mode = execution_mode
    self.graph_preprocess = graph_preprocess.copy()
    self.numerical_features = numerical_features.copy()
    self.categorical_features = categorical_features.copy()
    self.target = target_column
    self.artifacts = artifacts
    self.scaler = self.artifacts.get('scaler', None)
    self.oh_encoder = self.artifacts.get('oh_encoder', None)
    self.features_selected = self.artifacts.get('features_selected', None)


  def scale_features(self):
    """
    Scale the features of the DataFrame using the provided scaler.
    If the execution mode is 'train', the scaler is fitted on the training
    data, otherwise, it is applied to transform the data with the already
    fitted weights.
    """
    if self.execution_mode == 'train':
      logger.info('Fitting numerical scaler')
      self.scaler.fit(self.df[self.numerical_features])
      self.artifacts['scaler'] = self.scaler
    logger.info('Scaling numerical features')
    scaled_features = self.scaler.transform(self.df[self.numerical_features])
    self.df.loc[:, self.df.columns != self.target] = scaled_features
    logger.info('Completed scaling numerical features')


  def one_hot_encode(self):
    """
    Apply one-hot encoding to the categorical features of the DataFrame.
    If the execution mode is 'train', the encoder is fitted on the training
    data, otherwise, it is applied to transform the data with the already
    fitted weights.
    """
    if self.execution_mode == 'train':
      logger.info('Fitting one-hot encoder')
      self.oh_encoder.fit(self.df[self.categorical_features])
      self.artifacts['oh_encoder'] = self.oh_encoder
    logger.info('One-hot encoding categorical features')
    encoded_features = self.oh_encoder.transform(self.df\
                                                 [self.categorical_features])
    self.df = self.df.drop(self.categorical_features, axis=1)
    self.df = pd.concat([self.df, encoded_features], axis=1)
    logger.info('Completed one-hot encoding categorical features')
    logger.info('New number of features: %d',
                len(self.numerical_features + self.categorical_features)
                )


  def create_individual_features(self):
    """
    Create new features by applying different non-linear mathematical functions
    over all the original features. Each new feature will have a name
    given by the function name and the original feature name.
    """
    logger.info('Creating individual numerical features')
    list_functions, list_functions_names = get_default_math_functions()
    features_to_include = []
    for function, function_name in zip(list_functions, list_functions_names):
      logger.info(f'Creating {function_name} feature')
      for each_column in self.numerical_features:
        new_column = function_name + '_' + each_column
        self.df[new_column] = function(self.df[each_column])
        self.df[new_column] = self.df[new_column].fillna(0.0)
        features_to_include.append(new_column)
    self.numerical_features.extend(features_to_include)
    logger.info('Completed creating individual numerical features')
    logger.info('New number of features: %d',
                len(self.numerical_features + self.categorical_features)
                )


  def create_combinated_features(self):
    """
    Create new features by applying different mathematical operations over each
    pair of the original features provided. At the moment, the implemented
    features, for each combination of features x1 and x2 are: x1*x2, x1/x2 and
    x1%x2. The names of the new features will be given by x1_{name operation}_x2
    """
    logger.info('Creating combinated numerical features')
    features_to_include = []
    for feature_a in self.numerical_features:
      for feature_b in self.numerical_features:
        if feature_a != feature_b:
          logger.info(f'Creating features {feature_a}-{feature_b} interactions')
          new_column_1 = feature_a + '_prod_' + feature_b
          new_column_2 = feature_a + '_div_' + feature_b
          new_column_3 = feature_a + '_reminder_' + feature_b

          self.df[new_column_1] = self.df[feature_a]*self.df[feature_b]
          self.df[new_column_2] = self.df[feature_a]/(self.df[feature_b] + 1e-8)
          self.df[new_column_3] = self.df[feature_a]%self.df[feature_b]
          # As some function might create Null values, fill them with 0.0
          self.df[new_column_1] = self.df[new_column_1].fillna(0.0)
          self.df[new_column_2] = self.df[new_column_2].fillna(0.0)
          self.df[new_column_3] = self.df[new_column_3].fillna(0.0)
          features_to_include.extend([new_column_1, new_column_2, new_column_3])
    self.numerical_features.extend(features_to_include)
    logger.info('Completed creating combinated numerical features')
    logger.info('New number of features: %d',
                len(self.numerical_features + self.categorical_features)
                )


  def feature_selection(self):
    """
    Select the features to be used in the training process. If no features are
    provided, all the features will be used.
    """
    # TODO: Implement automatic feature selection methods
    logger.info('Selecting features')
    if self.features_selected is not None and self.execution_mode != 'test':
      self.df = self.df[self.features_selected + [self.target]]
    elif self.features_selected is not None and self.execution_mode == 'test':
      self.df = self.df[self.features_selected]
    self.numerical_features = [x for x in self.numerical_features\
                               if x in self.df.columns]
    self.categorical_features = [x for x in self.categorical_features\
                                 if x in self.df.columns]
    logger.info('Completed selecting features')
    logger.info('New number of features: %d',
                len(self.numerical_features + self.categorical_features)
                )


  def export_artifacts(self):
    """
    Use pickle to export the artifact objects to the artifacts folder.
    """
    logger.info('Exporting artifacts')
    export_files('../../artifacts', self.artifacts)
    logger.info('Completed exporting artifacts')


  def preprocess(self):
    """
    Entry point to apply all the defined preprocessing steps in a certain order.
    Returns:
      self.df: pd.DataFrame - Pandas DataFrame with the preprocessed data.
    """
    logger.info('Starting data preprocessing')
    logger.info('Initial number of features: %d',
                len(self.numerical_features + self.categorical_features)
                )
    if 'combinated_num' in self.graph_preprocess:
      self.create_combinated_features()
    if 'individual_num' in self.graph_preprocess:
      self.create_individual_features()
    if 'scale_num' in self.graph_preprocess and self.scaler is not None:
      self.scale_features()
    if 'oh_encode_cat' in self.graph_preprocess and self.oh_encoder is not None:
      self.one_hot_encode()
    
    self.feature_selection()
    if self.execution_mode == 'train':
      self.export_artifacts()

    logger.info('Completed data preprocessing')
    logger.info('Final number of features: %d',
                len(self.numerical_features + self.categorical_features)
                )
    return self.df
