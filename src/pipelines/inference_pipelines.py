from src.preprocessing.preprocessing import Preprocessor
from src.io.input import load_artifacts

class InferencePipeline:
  """
  Inference pipeline to use the model in production.
  """
  def __init__(self,
               model_path: str,
               artifacts_folder: str):
    self.artifacts = load_artifacts(artifacts_folder)
    self.model = load_artifacts(model_path)['model']
    self.scaler = self.artifacts['scaler']
    self.oh_encoder = self.artifacts['oh_encoder']
    self.num_features = self.artifacts['numerical_features']
    self.cat_features = self.artifacts['categorical_features']
    self.graph_preprocess = self.artifacts['graph_preprocess']

  def get_inference(self, df):
    self.preprocessor = Preprocessor(df,
                                     'prod',
                                     numerical_features=self.num_features,
                                     categorical_features=self.cat_features,
                                     target_column=None,
                                     scaler=self.scaler,
                                     oh_encoder=self.oh_encoder,
                                     graph_preprocess=self.graph_preprocess
                                    )
    self.df_preproc = self.preprocessor.preprocess()
    self.df_preproc = self.df_preproc[self.num_features + self.cat_features]
    
    return self.model.predict(self.df_preproc)
