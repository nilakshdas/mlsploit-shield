from datetime import datetime
import logging
import os
import sys
import time


def init_logger(name='main', logdir='./logs',
                log_to_stdout=True, log_to_file=True):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s\t[%(levelname)s]\t(%(module)s:'
        '%(lineno)03d)\t%(message)s')

    if log_to_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_to_file:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        assert os.path.isdir(logdir)

        fh = logging.FileHandler(os.path.join(logdir, '%s.log' % name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class CallTimer:
    def __enter__(self):
        self._start = time.clock()
        self._end = None
        return self

    def __exit__(self, *args):
        self._end = time.clock()

    @property
    def interval(self):
        return (time.clock() - self._start
                if self._end is None
                else self._end - self._start)
