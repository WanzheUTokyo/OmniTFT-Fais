
# Lint as: python3
"""Default configs for OmniTFT experiments.
"""

import os

# import data_formatter.temperature_formatter
# import data_formatter.bp_formatter
# import data_formatter.bun_formatter
# import data_formatter.hr_formatter
# import data_formatter.oxygen_saturation_formatter
import data_formatter.lactate_formatter
# import data_formatter.oxygenation_index_formatter
import data_formatter.creatinine_formatter
import data_formatter.respiratory_formatter


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
        # 'Temperature': data_formatter.temperature_formatter.TemperatureFormatter,
        # 'BP': data_formatter.bp_formatter.BPFormatter,
        # 'Bun': data_formatter.bun_formatter.BunFormatter,
        # 'HR': data_formatter.hr_formatter.HRFormatter,
        # 'OxygenSaturation': data_formatter.oxygen_saturation_formatter.OxygenSaturationFormatter,
        'Lactate': data_formatter.lactate_formatter.LactateFormatter,
        # 'OxygenationIndex': data_formatter.oxygenation_index_formatter.OxygenationIndexFormatter,
        'RespiratoryRate': data_formatter.respiratory_formatter.RespiratoryRateFormatter,
        'Creatinine': data_formatter.creatinine_formatter.CreatinineFormatter,

    }

    return data_formatter_class[self.experiment]()
