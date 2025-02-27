from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import Preprocessor
from src.io.input import load_artifacts
from src.hypertune.hypertuning import hyperparameter_tunning
from typing import Any
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class HyperTunnerPipeline:

    def __init__(self,
                    df: pd.DataFrame,
                    numerical_features: list,
                    categorical_features: list,
                    target_column: str,
                    preprocessing_graph: list,
                    artifacts: dict,
                    search_space: dict,
                    optimized_metric: str,
                    model2tune: Any,
                    model_name: str,
                    metrics_config: dict,
                    objective_score: Any,
                    test_size: float=0.2,
                    max_iterations: int=10
                    ):
            self.df = df.copy()
            self.num_features = numerical_features.copy()
            self.cat_features = categorical_features.copy()
            self.target_column = target_column
            self.preprocessing_graph = preprocessing_graph
            self.artifacts = artifacts
            self.search_space = search_space
            self.optimized_metric = optimized_metric
            self.model2tune = model2tune
            self.model_name = model_name
            self.metrics_config = metrics_config
            self.test_size = test_size
            self.max_iterations = max_iterations
            self.objective_score = objective_score


    def split_train_val(self):
        logger.info('Splitting data into train and validation sets')
        self.df_train, self.df_val = train_test_split(self.df,
                                                      test_size=self.test_size)

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


    def hypertune(self):
        logger.info('Starting hyperparameter tunning')
        self.best_hyperparams, self.trials = hyperparameter_tunning(
                                                        self.search_space,
                                                        self.max_iterations,
                                                        self.optimized_metric,
                                                        self.model2tune,
                                                        self.model_name,
                                                        self.df_train_preproc,
                                                        self.df_val_preproc,
                                                        self.target_column,
                                                        self.metrics_config,
                                                        self.objective_score
                                                        )
        logger.info('Completed hyperparameter tunning')

    def run(self):
        logger.info('Starting hyperparameter tunning pipeline')
        self.split_train_val()
        self.preprocess_data()
        self.hypertune()
        logger.info('Completed hyperparameter tunning pipeline')
