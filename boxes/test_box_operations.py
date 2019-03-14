import unittest
import torch
from box_operations import *

class TestDeltaBoxes(unittest.TestCase):

    def test_intersection_boxes(self):
        A = torch.tensor([[0.1, 0.2], [0.9, 0.5]])
        B = torch.tensor([[0.3, 0.1], [0.8, 1.0]])
        A_int_C = torch.tensor([[0.3, 0.2], [0.8, 0.5]])[None, None]
        A_and_B = torch.stack((A,B))[None,None]
        self.assertTrue((intersection(A_and_B) == A_int_C).all())

if __name__ == '__main__':
    unittest.main()

