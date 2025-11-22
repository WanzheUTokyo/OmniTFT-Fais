import abc
import enum


# Type defintions
"""Define some constants
REAL_VALUED: denotes real-valued data.
CATEGORICAL: denotes categorical data.
DATE: represents time-based data."""


class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""

    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


"""
TARGET: indicates a target variable (predicted value).
OBSERVED_INPUT: indicates observed input data.
KNOWN_INPUT: indicates known input data (known before model training).
STATIC_INPUT: indicates static inputs (features that do not change over time).
ID: indicates the identifier column, which is used to identify different entities.
TIME: indicates the time column, used for time indexing.
"""


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""

    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index


class GenericDataFormatter(abc.ABC):
    """Abstract base class for all data formatters.

    User can implement the abstract methods below to perform dataset-specific
    manipulations.

    """

    """This method is used to set the normaliser (Scaler) based on the given data (e.g. df). 
  The normalisation logic needs to be implemented in a specific subclass."""

    @abc.abstractmethod
    def set_scalers(self, df):
        """Calibrates scalers using the data supplied."""
        raise NotImplementedError()

    # Used to perform transformations on input data, such as normalisation or standardisation operations.
    @abc.abstractmethod
    def transform_inputs(self, df):
        """Performs feature transformation."""
        raise NotImplementedError()

    """The purpose of this method is to reverse transform the values predicted by the model 
  to the original scale before it was standardised or normalised."""

    @abc.abstractmethod
    def format_predictions(self, df):
        """Reverts any normalisation to give predictions in original scale."""
        raise NotImplementedError()

    # Implement a default division of the dataset into training, validation and test sets.
    @abc.abstractmethod
    def split_data(self, df):
        """Performs the default train, validation and test splits."""
        raise NotImplementedError()

    # Returns a column definition with column order, type (input/data type)
    @property
    @abc.abstractmethod
    def _column_definition(self):
        """Defines order, input type and data type of each column."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_fixed_params(self):
        """Defines the fixed parameters used by the model for training.

        Requires the following keys:
          'total_time_steps': Defines the total number of time steps used by TFT
          'num_encoder_steps': Determines length of LSTM encoder (i.e. history)
          'num_epochs': Maximum number of epochs for training
          'early_stopping_patience': Early stopping param for keras
          'multiprocessing_workers': # of cpus for data processing


        Returns:
          A dictionary of fixed parameters, e.g.:

          fixed_params = {
              'total_time_steps': 252 + 5,
              'num_encoder_steps': 252,
              'num_epochs': 100,
              'early_stopping_patience': 5,
              'multiprocessing_workers': 5,
          }
        """
        raise NotImplementedError

    # Shared functions across data-formatters,
    # These functions do not require subclass implementation and can be used directly by generic data processors.
    # Returns the number of categories per category input, typically used in the Keras embedding layer.
    @property
    def num_classes_per_cat_input(self):
        """Returns number of categories per relevant input.

        This is seqeuently required for keras embedding layers.
        """
        return self._num_classes_per_cat_input

    # Returns the number of samples used for training and validation. -1 means use all available samples.
    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.

        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.

        Returns:
          Tuple of (training samples, validation samples)
        """
        return -1, -1

    # Returns a definition of the columns in the order expected by the TFT model and ensures that the ID and TIME columns are unique.
    def get_column_definition(self):
        """ "Returns formatted column definition in order expected by the TFT."""

        column_definition = self._column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        def _check_single_column(input_type):

            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError(
                    "Illegal number of inputs ({}) of type {}".format(
                        length, input_type
                    )
                )

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.REAL_VALUED
            and tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]
        categorical_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.CATEGORICAL
            and tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        return identifier + time + real_inputs + categorical_inputs

    # Returns all column names used for input, excluding the ID and TIME columns.
    def _get_input_columns(self):
        """Returns names of all input columns."""
        return [
            tup[0]
            for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

    # Returns details of the input indexes and inputs associated with the TFT model.
    def _get_tft_input_indices(self):
        """Returns the relevant indexes and input sizes required by TFT."""

        # Functions
        def _extract_tuples_from_data_type(data_type, defn):
            return [
                tup
                for tup in defn
                if tup[1] == data_type
                and tup[2] not in {InputTypes.ID, InputTypes.TIME}
            ]

        def _get_locations(input_types, defn):
            return [i for i, tup in enumerate(defn) if tup[2] in input_types]

        # Start extraction
        column_definition = [
            tup
            for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        categorical_inputs = _extract_tuples_from_data_type(
            DataTypes.CATEGORICAL, column_definition
        )
        real_inputs = _extract_tuples_from_data_type(
            DataTypes.REAL_VALUED, column_definition
        )

        locations = {
            "input_size": len(self._get_input_columns()),
            "output_size": len(_get_locations({InputTypes.TARGET}, column_definition)),
            "category_counts": self.num_classes_per_cat_input,
            "input_obs_loc": _get_locations({InputTypes.TARGET}, column_definition),
            "static_input_loc": _get_locations(
                {InputTypes.STATIC_INPUT}, column_definition
            ),
            "known_regular_inputs": _get_locations(
                {InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, real_inputs
            ),
            "known_categorical_inputs": _get_locations(
                {InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, categorical_inputs
            ),
        }

        return locations

    """Returns the fixed parameters needed for the experiment, 
  checks that the necessary fields exist, and includes the column definitions and input indexes."""

    def get_experiment_params(self):
        """Returns fixed model parameters for experiments."""

        required_keys = [
            "total_time_steps",
            "num_encoder_steps",
            "num_epochs",
            "early_stopping_patience",
            "multiprocessing_workers",
        ]

        fixed_params = self.get_fixed_params()

        for k in required_keys:
            if k not in fixed_params:
                raise ValueError(
                    "Field {}".format(k) + " missing from fixed parameter definitions!"
                )

        fixed_params["column_definition"] = self.get_column_definition()

        fixed_params.update(self._get_tft_input_indices())

        return fixed_params
