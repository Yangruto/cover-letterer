import logging

NOISY_LOGGERS = [
    "httpcore",
    "httpx",
    "urllib3",
    "requests",
    "asyncio",
    "PIL",
    "matplotlib",
    "h5py",
    "tensorflow",
]


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


setup_logging()
