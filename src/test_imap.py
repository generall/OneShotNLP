import time
from multiprocessing.pool import Pool
import random

import logging
import os
import multiprocessing
from time import sleep

from threading import Semaphore

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(process)d:%(thread)d:%(message)s')
logger = logging.getLogger()


def random_generator(semaphore):
    while True:
        semaphore.acquire()
        yield random.randint(1, 10)


def plus(x):
    return x + 1


def pooling():
    with Pool(2) as pool:
        semaphore = Semaphore(10)
        for x in pool.imap_unordered(plus, random_generator(semaphore)):
            yield x
            semaphore.release()


for x in pooling():
    time.sleep(1)
    print(x)
