import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import Preprocessor
from src.io.input import load_artifacts
from src.train.training import train_multiple_models


class StandardTrainerPipeline:
    """
    A class to represent a standard training pipeline.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 numerical_features: list,
                 categorical_features: list,
                 target_column: str,
                 preprocessing_graph: list,
                 artifacts: dict,
                 list_models: list,
                 list_model_names: list,
                 metrics_config: dict,
                 test_size: float=0.2
                ):
        self.df = df.copy()
        self.num_features = numerical_features.copy()
        self.cat_features = categorical_features.copy()
        self.target_column = target_column
        self.preprocessing_graph = preprocessing_graph
        self.artifacts = artifacts
        self.list_models = list_models
        self.list_model_names = list_model_names
        self.metrics_config = metrics_config
        self.test_size = test_size

    def split_train_val(self):
        self.df_train, self.df_val = train_test_split(self.df,
                                                      test_size=self.test_size)

    def preprocess_data(self):
        self.train_pipeline = Preprocessor(
                                    self.df_train,
                                    'train',
                                    numerical_features=self.num_features,
                                    categorical_features=self.cat_features,
                                    target_column=self.target_column,
                                    graph_preprocess=self.preprocessing_graph,
                                    artifacts=self.artifacts
                                    )
        self.df_train_preproc = self.train_pipeline.preprocess()

        # Load generated artifacts
        self.artifacts = load_artifacts('../../artifacts/')
        # Apply same preprocess logic to validation set
        self.validation_pipeline = Preprocessor(
                                    self.df_val,
                                    'val',
                                    numerical_features=self.num_features,
                                    categorical_features=self.cat_features,
                                    target_column=self.target_column,
                                    graph_preprocess=self.preprocessing_graph,
                                    artifacts=self.artifacts
                                    )
        self.df_val_preproc = self.validation_pipeline.preprocess()
    
    def train_models(self):
        self.metrics_df, self.fitted_models = train_multiple_models(
                                                    self.df_train_preproc,
                                                    self.df_val_preproc,
                                                    self.list_models,
                                                    self.list_model_names,
                                                    self.target_column,
                                                    self.metrics_config
                                                    )

    def run(self):
        self.split_train_val()
        self.preprocess_data()
        self.train_models()
