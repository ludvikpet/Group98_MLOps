import pandas as pd
from sklearn import datasets
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

@hydra.main(config_path=to_absolute_path("configs"), config_name="config", version_base="1.1")
def data_drift(cfg: DictConfig) -> None:

  reference_data = datasets.load_iris(as_frame=True).frame
  reference_data = reference_data.rename(
      columns={
          'sepal length (cm)': 'sepal_length',
          'sepal width (cm)': 'sepal_width',
          'petal length (cm)': 'petal_length',
          'petal width (cm)': 'petal_width',
          'target': 'target'
      }
  )

  current_data = pd.read_csv('prediction_database.csv')
  current_data = current_data.drop(columns=['time'])

  report = Report(metrics=[DataDriftPreset(),DataQualityPreset(),TargetDriftPreset()])
  report.run(reference_data=reference_data, current_data=current_data)
  report.save_html('report.html')

if __name__ == "__main__":
  data_drift()
