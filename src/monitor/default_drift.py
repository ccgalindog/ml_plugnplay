from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def get_default_drift_report(df_ref: pd.DataFrame,
                             df_current: pd.DataFrame):
    """
    Get the default drift report.
    Returns:
      report: Report - the default drift report.
    """
    logger.info('Creating default drift report')
    report = Report(metrics=[DataDriftPreset()])
    report.run(current_data=df_ref, reference_data=df_current)
    logger.info('Completed drift report')
    return report
