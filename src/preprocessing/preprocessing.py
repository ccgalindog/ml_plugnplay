import pandas as pd
from typing import Any
import numpy as np
from io.output import export_files


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
               scaler: Any=None,
               oh_encoder: Any=None):
    """
    Preprocessor constructor.
    Args:
      df: pd.DataFrame - Pandas DataFrame with the data to be preprocessed.
      execution_mode: str - Flag that defines the type of execution, either
        'train', 'validation' or 'test'.
      scaler: Any - Scaler object from sklearn.
      oh_encoder: Any - One-hot encoder object from sklearn.
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
    """
    self.df = df.copy()
    self.execution_mode = execution_mode
    self.graph_preprocess = graph_preprocess
    self.scaler = scaler
    self.oh_encoder = oh_encoder
    self.numerical_features = numerical_features
    self.categorical_features = categorical_features
    self.target = target_column
    self.artifacts = {}
  
  def scale_features(self):
    """
    Scale the features of the DataFrame using the provided scaler.
    If the execution mode is 'train', the scaler is fitted on the training
    data, otherwise, it is applied to transform the data with the already
    fitted weights.
    """
    if self.execution_mode == 'train':
      self.scaler.fit(self.df[self.numerical_features])
    scaled_features = self.scaler.transform(self.df[self.numerical_features])
    self.df.loc[:, self.df.columns != self.target] = scaled_features
    self.artifacts['scaler'] = self.scaler

  def one_hot_encode(self):
    """
    Apply one-hot encoding to the categorical features of the DataFrame.
    If the execution mode is 'train', the encoder is fitted on the training
    data, otherwise, it is applied to transform the data with the already
    fitted weights.
    """
    if self.execution_mode == 'train':
      self.oh_encoder.fit(self.df[self.categorical_features])
    encoded_features = self.oh_encoder.transform(self.df\
                                                 [self.categorical_features])
    self.df = self.df.drop(self.categorical_features, axis=1)
    self.df = pd.concat([self.df, encoded_features], axis=1)
    self.artifacts['oh_encoder'] = self.oh_encoder

  def create_individual_features(self):
    """
    Create new features by applying different non-linear mathematical functions
    over all the original features. Each new feature will have a name
    given by the function name and the original feature name.
    """
    list_functions = [np.log, np.log10, np.sqrt, np.square,
                      np.arcsin, np.arccos, np.arctan,
                      np.sin, np.cos, np.tan, np.sinc]
    list_functions_names = ['log', 'log10', 'sqrt', 'square',
                            'arcsin', 'arccos', 'arctan',
                            'sin', 'cos', 'tan', 'sinc']
    features_to_include = []
    for function, function_name in zip(list_functions, list_functions_names):
      #print(f'Creating {function_name} feature')
      for each_column in self.numerical_features:
        new_column = function_name + '_' + each_column
        self.df[new_column] = function(self.df[each_column])
        self.df[new_column] = self.df[new_column].fillna(0.0)
        features_to_include.append(new_column)
    self.numerical_features.extend(features_to_include)

  def create_combinated_features(self):
    """
    Create new features by applying different mathematical operations over each
    pair of the original features provided. At the moment, the implemented
    features, for each combination of features x1 and x2 are: x1*x2, x1/x2 and
    x1%x2. The names of the new features will be given by x1_{name operation}_x2
    """
    features_to_include = []
    for feature_a in self.numerical_features:
      for feature_b in self.numerical_features:
        if feature_a != feature_b:
          #print(f'Creating features from {feature_a}-{feature_b} interactions')
          new_column_1 = feature_a + '_prod_' + feature_b
          new_column_2 = feature_a + '_div_' + feature_b
          new_column_3 = feature_a + '_reminder_' + feature_b

          self.df[new_column_1] = self.df[feature_a]*self.df[feature_b]
          self.df[new_column_2] = self.df[feature_a]/self.df[feature_b]
          self.df[new_column_3] = self.df[feature_a]%self.df[feature_b]
          # As some function might create Null values, fill them with 0.0
          self.df[new_column_1] = self.df[new_column_1].fillna(0.0)
          self.df[new_column_2] = self.df[new_column_2].fillna(0.0)
          self.df[new_column_3] = self.df[new_column_3].fillna(0.0)
          features_to_include.extend([new_column_1, new_column_2, new_column_3])
    self.numerical_features.extend(features_to_include)

  def export_artifacts(self):
    """
    Use pickle to export the artifact objects to the artifacts folder.
    """
    export_files('artifacts', self.artifacts)

  def preprocess(self):
    """
    Entry point to apply all the defined preprocessing steps in a certain order.
    Returns:
      self.df: pd.DataFrame - Pandas DataFrame with the preprocessed data.
    """
    if 'individual_num' in self.graph_preprocess:
      self.create_individual_features()
    if 'combinated_num' in self.graph_preprocess:
      self.create_combinated_features()
    if 'scale_num' in self.graph_preprocess and self.scaler is not None:
      self.scale_features()
    if 'oh_encode_cat' in self.graph_preprocess and self.oh_encoder is not None:
      self.one_hot_encode()
    if self.execution_mode == 'train':
      self.export_artifacts()
    return self.df
