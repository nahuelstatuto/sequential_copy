import unittest
from models import FeedForwardModel
from sampling import Sampler


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    sample = Sampler()
    
    # test case function to check the Sampler.set_name function
    def test_0_open_file_for_read(self):
        print("Start open_file_for_read test\n")
        sample = Sampler()
        test_return = sample.open_file_for_read()
        self.assertIsNone(test_return)
        self.assertFalse(sample.from_file)
        self.assertFalse(sample.to_file)
        
        sample = Sampler()
        sample.from_file = True
        test_return = sample.open_file_for_read()
        self.assertIsNone(test_return)
        self.assertFalse(sample.from_file)
        self.assertFalse(sample.to_file)
        
        sample = Sampler()
        sample.from_file = True
        sample.automatic_fill = True
        test_return = sample.open_file_for_read()
        self.assertIsNone(test_return)
        self.assertFalse(sample.from_file)
        self.assertFalse(sample.to_file)
        
        sample = Sampler()
        sample.file_path = 'test.csv'
        sample.from_file = True
        test_return = sample.open_file_for_read()
        self.assertIsNotNone(test_return)
        self.assertTrue(sample.from_file)
        test_return.close()
        
        
        
        print("\nFinish open_file_for_read test\n")

    


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()