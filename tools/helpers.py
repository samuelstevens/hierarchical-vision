import collections
import logging
import pathlib
import statistics
import sys


class ClassDistribution:
    def __init__(self, seq):
        self.counts = collections.Counter(seq)

    def min(self):
        return self.counts.most_common()[-1]

    def max(self):
        return self.counts.most_common(1)[0]

    def mean(self):
        return statistics.mean(self.counts.values())


def create_logger(name, output_dir: pathlib.Path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = "[%(asctime)s] [%(filename)s:%(lineno)d %(levelname)s] %(message)s"
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "log.txt", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.info("Made directory %s", output_dir)

    return logger
