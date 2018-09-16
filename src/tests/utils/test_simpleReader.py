from unittest import TestCase

from utils.loader import SimpleReader


class TestSimpleReader(TestCase):
    def test_read_batches(self):
        reader = SimpleReader('/home/generall/sources/hackathones/tele2/data/valid_data.tsv', batch_size=100)

        for batch in reader.read_batches():
            q, a, m = batch

            print(a)
            break
