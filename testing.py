import os
import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

from sampling import Sampler
from models import FeedForwardModel
from models import params_to_vec
from utils import define_loss
from utils import LambdaParameter

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


class TestFeedForwardModel(unittest.TestCase):
    input_dim = 10
    output_dim = 2
    
    def setUp(self):
        layers = [20,30]
        self.model = FeedForwardModel(input_dim=self.input_dim, hidden_layers=layers, output_dim=self.output_dim)
        self.model.build(input_shape=(layers[0],self.input_dim))
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.loss = define_loss(self.output_dim, loss_name = 'UncertaintyError')
        self.model.compile(loss=self.loss, optimizer=self.opt)
        self.model.theta0, self.model.weights_dims = params_to_vec(self.model, return_dims=True)
        
    def test_call(self):
        x = tf.random.normal((32, self.input_dim))
        y_pred = self.model(x)
        self.assertEqual(y_pred.shape, (32, self.output_dim))
        
    def test_fit(self):
        x = tf.random.normal((100, self.input_dim))
        y = tf.one_hot(tf.random.uniform((100,), maxval=2, dtype=tf.int32), depth=2)
        lmda = 0.01
        rho_max = 1.0
        epochs = 5
        batch_size = 32
        history = self.model.fit(x, y, lmda=lmda, rho_max=rho_max, epochs=epochs, batch_size=batch_size, verbose=0)
        self.assertEqual(len(history.history['loss']), epochs)
        self.assertGreater(history.history['loss'][0], history.history['loss'][-1])
        
    def test_train_step(self):
        x = tf.random.normal((100, self.input_dim))
        y = self.model(x)
        self.model.lmda = 0.01
        self.model.rho_max = 1.0
        data = (x, y)
        logs = self.model.train_step(data)
        self.assertTrue('loss' in logs)
        self.assertTrue('reg' in logs)
        self.assertTrue('rho' in logs)
        self.assertIsInstance(logs['loss'], tf.Tensor)
        self.assertIsInstance(logs['reg'], tf.Tensor)
        self.assertIsInstance(logs['rho'], tf.Tensor)

        
class TestLambdaParameter(unittest.TestCase):
    
    def test_lmda_manual(self):
        lmda = 0.1
        lambda_param = LambdaParameter(lmda=lmda)
        self.assertEqual(lambda_param.lmda, lmda)
        lambda_param.lmda = 0.5
        self.assertEqual(lambda_param.lmda, 0.5)
        
    def test_lmda_automatic(self):
        nN_prev, nN, n_sampling = 100, 90, 15
        
        lmda = 0.1
        divider=2
        multiplier=1.5
        
        lambda_param = LambdaParameter(lmda=lmda, automatic_lmda=True, divider=divider, multiplier=multiplier)
        
        lambda_param.update(nN_prev, nN, n_sampling)
        lmda = lmda/divider
        self.assertAlmostEqual(lambda_param.lmda, lmda, delta=0.001)

        lambda_param.update(nN_prev, nN, n_sampling)
        lmda = lmda/divider
        self.assertAlmostEqual(lambda_param.lmda, lmda, delta=0.001)
        
        n_sampling = 5
        
        lambda_param.update(nN_prev, nN, n_sampling)
        lmda = lmda*multiplier
        self.assertAlmostEqual(lambda_param.lmda, lmda, delta=0.001)

        
        lambda_param.update(nN_prev, nN, n_sampling)
        lmda = lmda*multiplier
        self.assertAlmostEqual(lambda_param.lmda, lmda, delta=0.001)
        
if __name__ == '__main__':
    unittest.main()