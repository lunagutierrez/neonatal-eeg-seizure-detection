import logging

def get_logger(name: str): # name = module so it can be traced back
    """
    Creates and returns a logger instance with a consistent format.
    Returns: logging.Logger a Configured logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()  # Writes log messages to the console
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger