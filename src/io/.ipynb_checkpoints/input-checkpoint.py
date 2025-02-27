import pickle
from glob import glob

def load_artifacts(artifacts_folder):
  """
  Load all the artifacts from the artifacts folder.
  Args:
    artifacts_folder: str - Path to the folder containing the artifacts.
  Returns: 
    artifacts: Dict - A dictionary containing all the artifacts
  """
  list_artifacts = glob(f'{artifacts_folder}/*.pkl')
  artifacts = {}
  for artifact_file in list_artifacts:
    with open(artifact_file, 'rb') as f:
      artifacts[artifact_file.split('/')[-1].split('.')[0]] = pickle.load(f)
  return artifacts
