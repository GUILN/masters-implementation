from logging import INFO, ERROR

from application_logger import ApplicationLogger



if __name__ == "__main__":
    # Example usage
    logger = ApplicationLogger(
        loglevel=INFO,
        logfile="example.log",
        log_to_stdout=True,
        sentry_dsn=None
    )

    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("An exception occurred!")
