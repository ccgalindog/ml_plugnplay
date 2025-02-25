from src.preprocessing.preprocessing import Preprocessor
from src.io.input import load_artifacts
import pickle

class InferencePipeline:
  """
  Inference pipeline to use the model in production.
  """
  # TODO: Modify for execution graph instead of create_features
  def __init__(self,
               model_path: str,
               artifacts_folder: str,
               create_features: str):
    artifacts = load_artifacts(artifacts_folder)
    self.create_features = create_features

  def get_inference(self, df):
    self.preprocessor = Preprocessor(df,
                                     'prod',
                                      scaler=self.scaler,
                                      create_features=self.create_features
                                     )
    self.df_preproc = self.preprocessor.preprocess()
    if self.selected_features is not None:
      self.df_preproc = self.df_preproc[self.selected_features]
    
    return self.model.predict(self.df_preproc)
