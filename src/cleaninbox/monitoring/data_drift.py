import sys
import pandas as pd
from sklearn import datasets
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from google.cloud.storage import Bucket
import io

from google.cloud import storage
from google.cloud.storage import Bucket

import io

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

from cleaninbox.data import get_monitoring_data

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{message}</green> | {level} | {time:HH:mm:ss}")

def data_drift(cfg: DictConfig, bucket: Bucket = None) -> None:

  if bucket:
    logger.info("Retrieving reference data and new data from gcs...")
    # Retrieve monitoring data from GCS:
    reference_data, new_data = get_monitoring_data(bucket, cfg.gs.monitoring)

    logger.info("Done retrieving data, preparing data for Evidently...")
    # file.write("time, model_name, input_length, target, prediction_time\n")
    new_data = new_data.drop(columns=['time', 'model_name', 'prediction_time'])
    new_data = new_data.rename(
        columns={
            'prediction': 'target'
        }
    )

    logger.info("Done preparing data, running Evidently...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_data, current_data=new_data)

    report_html = io.StringIO()
    report.save_html(report_html)
    report_html.seek(0)
    bucket.blob(f"{cfg.gs.monitoring}reports/report.html").upload_from_string(report_html.getvalue(), content_type='text/html')
    logger.info("Done running Evidently, report saved to GCS.")

    return
  else:

    logger.info("Retrieving reference data and new data locally...")

    reference_data, new_data = get_monitoring_data(None, cfg.gs.monitoring)

    logger.info("Done retrieving data...")
    #logger.info(f"Newdata shape: {new_data.shape}\tNewdata columns: {new_data.columns}")
    #logger.info(f"Referencedata shape: {reference_data.shape}\tReferencedata columns: {reference_data.columns}")

    # file.write("time, model_name, input_length, target, prediction_time\n")
    new_data = new_data.drop(columns=['time', 'model_name', 'prediction_time'])
    new_data = new_data.rename(
        columns={
            'prediction': 'target'
        }
    )

    logger.info("Done preparing data, running Evidently...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_data, current_data=new_data)

    report.save_html("report.html")
    logger.info("Done running Evidently, report saved locally.")

@hydra.main(config_path=to_absolute_path("configs"), config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:

  storage_client = storage.Client()
  bucket = storage_client.bucket(cfg.gs.bucket)

  data_drift(cfg, bucket)

if __name__ == "__main__":
    main()

