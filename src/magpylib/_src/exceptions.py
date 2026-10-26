"""Definition of custom exceptions"""


class MagpylibBadUserInput(ValueError):
    """bad user input"""


class MagpylibInternalError(RuntimeError):
    """should never have reached this position in the code"""


class MagpylibBadInputShape(ValueError):
    """catching bad input shapes"""


class MagpylibMissingInput(ValueError):
    """catching missing user inputs"""


# removed MagpylibDeprecationWarning - there is a built-in DeprecationWarning
# other will be removed soon sticking to standards
