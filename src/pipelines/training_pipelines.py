import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import Preprocessor
from src.io.input import load_artifacts
from src.train.training import train_multiple_models
import logging
logger = logging.getLogger(__name__)


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
        logger.info('Splitting data into train and validation sets')
        self.df_train, self.df_val = train_test_split(self.df,
                                                      test_size=self.test_size)
        logger.info('Completed data split')

    def preprocess_data(self):
        logger.info('Starting data preprocessing')
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

        self.artifacts = load_artifacts('../../artifacts/')

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
        logger.info('Completed data preprocessing')
    
    def train_models(self):
        logger.info('Starting model training')
        self.metrics_df, self.fitted_models = train_multiple_models(
                                                    self.df_train_preproc,
                                                    self.df_val_preproc,
                                                    self.list_models,
                                                    self.list_model_names,
                                                    self.target_column,
                                                    self.metrics_config
                                                    )
        logger.info('Completed model training')

    def run(self):
        logger.info('Starting training pipeline')
        self.split_train_val()
        self.preprocess_data()
        self.train_models()
        logger.info('Completed training pipeline')
