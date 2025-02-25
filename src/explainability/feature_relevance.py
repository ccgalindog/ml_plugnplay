from typing import Any, List, Tuple
import numpy as np
import plotly.graph_objects as go


def plot_feature_importance(model: Any,
                            feature_names: List,
                            limit_features: int=40) -> Tuple[Any, List]:
  """
  Compute feature importance and plot it. Only the main features are plotted,
  the number is defined by the limit_features parameter. Those most relevant
  features are also returned.
  Args:
    model: Any - Model object / estimator.
    feature_names: List - List of feature names.
    limit_features: int - Number of most relevant features to plot.
  Returns:
    fig: go.Figure - Plotly figure with the feature importance.
    lim_feature_names: List - List of the most relevant features.
  """
  importances = model.feature_importances_

  sorter = np.argsort(importances) 
  lim_feature_names = np.array(feature_names)[sorter][-limit_features:]
  lim_importances = importances[sorter][-limit_features:]
  
  fig = go.Figure()
  fig.add_trace(go.Bar(name='Feature',
                      x=lim_feature_names, y=lim_importances
                      )
                )
  return fig, lim_feature_names
