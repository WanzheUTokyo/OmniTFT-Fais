
# Lint as: python3
"""Default configs for TFT experiments.

Contains the default output paths for data, serialised models and predictions
for the main experiments used in the publication.
"""

import os

# import data_formatters.Temperature
# import data_formatters.BP
# import data_formatters.Bun
# import data_formatters.HR
# import data_formatters.OxygenSaturation
# import data_formatters.Lactate
# import data_formatters.OxygenationIndex
import data_formatters.Creatinine
import data_formatters.RespiratoryRate


class ExperimentConfig(object):
      # 'Temperature',
      # 'BP',
      # 'Bun',
      # 'HR',
      # 'OxygenSaturation',
      # 'Lactate',
      # 'OxygenationIndex',

  default_experiments = [
      'Lactate',
      'RespiratoryRate',
      'Creatinine',
  ]

  def __init__(self, experiment='Temperature', root_folder=None):
    """Creates configs based on default experiment chosen.

    Args:
      experiment: Name of experiment.
      root_folder: Root folder to save all outputs of training.
    """

    if experiment not in self.default_experiments:
      raise ValueError('Unrecognised experiment={}'.format(experiment))

    # Defines all relevant paths
    if root_folder is None:
      root_folder = os.path.join(
          os.path.dirname(os.path.realpath(__file__)), '..', 'outputs')
      print('Using root folder {}'.format(root_folder))

    self.root_folder = root_folder
    self.experiment = experiment
    self.data_folder = os.path.join(root_folder, 'data', experiment)
    self.model_folder = os.path.join(root_folder, 'saved_models', experiment)
    self.results_folder = os.path.join(root_folder, 'results', experiment)

    # Creates folders if they don't exist
    for relevant_directory in [
        self.root_folder, self.data_folder, self.model_folder,
        self.results_folder
    ]:
      if not os.path.exists(relevant_directory):
        os.makedirs(relevant_directory)
  
  # The trained file is saved here, if we download the train data with api, the output should be here
  @property
  def data_csv_path(self):
    csv_map = {
        'Temperature': 'Temperature.csv',
        'BP': 'BP.csv',
        'Bun': 'Bun.csv',
        'HR': 'HR.csv',
        'OxygenSaturation': 'OxygenSaturation.csv',
        'Lactate': 'lactate.csv',
        'OxygenationIndex': 'OxygenationIndex.csv',
        'RespiratoryRate': 'RespiratoryRate.csv',
        'Creatinine': 'creatinine.csv',

    }

    return os.path.join(self.data_folder, csv_map[self.experiment])

  @property
  def hyperparam_iterations(self):

    return 30

  def make_data_formatter(self):
    """Gets a data formatter object for experiment.

    Returns:
      Default DataFormatter per experiment.
    """

    data_formatter_class = {
        # 'Temperature': data_formatters.Temperature.TemperatureFormatter,
        # 'BP': data_formatters.BP.BPFormatter,
        # 'Bun': data_formatters.Bun.BunFormatter,
        # 'HR': data_formatters.HR.HRFormatter,
        # 'OxygenSaturation': data_formatters.OxygenSaturation.OxygenSaturationFormatter,
        'Lactate': data_formatters.Lactate.LactateFormatter,
        # 'OxygenationIndex': data_formatters.OxygenationIndex.OxygenationIndexFormatter,
        'RespiratoryRate': data_formatters.RespiratoryRate.RespiratoryRateFormatter,
        'Creatinine': data_formatters.Creatinine.CreatinineFormatter,

    }

    return data_formatter_class[self.experiment]()
