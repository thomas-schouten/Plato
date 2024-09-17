import setup

class Settings():
    def __init__(
            self,
            _name,
            _ages,
            _cases,
            _options,
            _dir,
            PARALLEL_MODE = False,
            DEBUG_MODE = False,
        ):
        """
        Object to store the settings of the plato simulation
        """
        self.name = _name
        self.ages = _ages
        self.cases = _cases
        self.options = _options
        self.dir = _dir
        self.PARALLEL_MODE = PARALLEL_MODE
        self.DEBUG_MODE = DEBUG_MODE

        # Group cases to accelerate initialisation of plate data
        plate_options = ["Minimum plate area"]
        self.plate_cases = setup.process_cases(self.cases, self.options, plate_options)