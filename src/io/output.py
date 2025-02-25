import pickle
from typing import Dict


def export_files(folder_name: str, artifacts_dict: Dict):
  """
  Use pickle to export the artifact objects to the artifacts folder.
  Args:
    artifacts: Dict - A dictionary containing all the artifacts
  """
  for artifact_name, artifact in artifacts_dict.items():
    with open(f'{folder_name}/{artifact_name}.pkl', 'wb') as f:
      pickle.dump(artifact, f)
