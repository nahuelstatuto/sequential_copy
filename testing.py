import os
import unittest
import numpy as np
from sampling import Sampler
from sklearn.ensemble import RandomForestClassifier

class SamplerTest(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler()
        self.original = self.get_original_model()

    def test_generate_samples(self):
        X, y = self.sampler.generate_samples(self.original, 100)
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(y.shape, (100,))
        
    def test_add_samples_to_file(self):
        self.sampler.file_path = 'test.txt'
        X = np.random.rand(100, 2)
        y = np.random.randint(0, 2, size=100)
        self.sampler.add_samples_to_file(X, y)
        self.sampler.file.close()
        with open(self.sampler.file_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 100)
            for line in lines:
                self.assertEqual(len(line.split(',')), 3)

    def test_read_samples_from_file(self):
        self.sampler.file_path = 'test.txt'
        X = np.random.rand(200, 2)
        y = np.random.randint(0, 2, size=200)
        with open(self.sampler.file_path, 'w') as f:
            for i in range(200):
                f.write(f"{X[i, 0]},{X[i, 1]},{y[i]}\n")
        self.sampler.from_file = True
        self.sampler.set_file()
        X_new, y_new = self.sampler.read_samples_from_file(self.original, 100)
        self.assertEqual(X_new.shape, (100, 2))
        self.assertEqual(y_new.shape, (100,))
        self.assertEqual(self.sampler.iteration, 1)

    def test_set_file(self):
        self.sampler.from_file = True
        self.sampler.file_path = 'non_existent_file.txt'
        self.sampler.set_file()
        self.assertFalse(self.sampler.from_file)
        self.assertFalse(self.sampler.to_file)
        self.assertFalse(self.sampler.automatic_fill)
        self.assertIsNone(self.sampler.file)
        
        self.sampler.from_file = True
        self.sampler.file_path = None
        self.sampler.automatic_fill = False
        self.sampler.set_file()
        self.assertFalse(self.sampler.from_file)
        self.assertFalse(self.sampler.to_file)
        self.assertFalse(self.sampler.automatic_fill)
        self.assertIsNone(self.sampler.file)
        
        self.sampler.from_file = True
        self.sampler.file_path = 'non_existent_file.txt'
        self.sampler.automatic_fill = True
        self.sampler.set_file()
        self.assertFalse(self.sampler.from_file)
        self.assertTrue(self.sampler.to_file)
        self.assertTrue(self.sampler.automatic_fill)
        self.assertIsNotNone(self.sampler.file)
        os.remove(self.sampler.file_path)
        
        self.sampler.from_file = True
        self.sampler.file_path = 'test.txt'
        self.sampler.set_file()
        self.assertTrue(self.sampler.from_file)
        self.assertTrue(self.sampler.to_file)
        self.assertTrue(self.sampler.automatic_fill)
        self.assertIsNotNone(self.sampler.file)
        self.sampler.file.close()
        os.remove(self.sampler.file_path)

    def tearDown(self):
        if self.sampler.file:
            self.sampler.file.close()
            self.sampler.file = None
        
    def get_original_model(self):
        original = RandomForestClassifier(n_estimators=1, criterion='gini', n_jobs=1, max_depth=None, random_state=42)
        original.fit([[0.9,0.35]], [1.0])
        return original

if __name__ == '__main__':
    unittest.main()