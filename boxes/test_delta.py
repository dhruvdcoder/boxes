import unittest
from boxes import *

class TestDeltaBoxes(unittest.TestCase):

    def test_create_boxes(self):
        boxes = DeltaBoxes(1,4,2)


if __name__ == '__main__':
    unittest.main()