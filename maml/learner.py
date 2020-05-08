import matplotlib.pyplot as plt
import tensorflow as tf
from maml.data_handler import data_gen, Sinusoid
from maml.model import SineModel


class Learner:
    def __init__(self, epochs=1, lr_inner=0.01, batch_size=1, log_steps=1000):
        self.epochs = epochs
        self.train_dataset, self.test_dataset = data_gen(K=10)
        self.dummy = tf.constant(self.train_dataset[0].batch()[0])
        self.lr_inner = lr_inner
        self.batch_size = batch_size
        self.log_steps = log_steps
        self._new_model()

    def _new_model(self):
        self.model = SineModel()
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def clone_model(self):
        cloned = SineModel()
        cloned(self.dummy)
        cloned.set_weights(self.model.get_weights())
        return cloned

    def compute_loss(self, model, x, y):
        logits = model(x)
        loss = self.loss_fn(y, logits)
        return loss

    def train_vanilla(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(self.model, x, y)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def _learn(self, train_fn, name):
        print(name)
        for e in range(self.epochs):
            losses = []
            total_loss = 0.
            for i, sinusoid in enumerate(self.train_dataset):
                slices = (tf.constant(z) for z in sinusoid.batch())
                loss = train_fn(*slices)
                total_loss += loss.numpy()
                losses.append(total_loss/(i+1))
                if i % self.log_steps == 0 and i > 0:
                    print(f'Epoch: {e} | Step: {i} | Loss: {total_loss/(i+1):.5f}')
        plt.plot(losses)
        plt.title(f'{name} loss curve')
        plt.savefig(f'images/{name}_loss.png')
        plt.close()

    def _evaluate(self, name):
        generator = Sinusoid(K=10)
        x_test, y_test = generator.equally_spaced_samples(100)
        x, y = generator.batch()
        cloned = self.clone_model()
        optim = tf.keras.optimizers.SGD(0.01)

        train, = plt.plot(x, y, '^')
        truth, = plt.plot(x_test, y_test)
        plots = [train, truth]
        legends = ['Training Points', 'True Function']
        for i in range(11):
            if i in [0, 1, 10]:
                pred = cloned(x_test)
                step, = plt.plot(x_test, pred, '--')
                plots.append(step)
                legends.append(f'After {i} Step')
            with tf.GradientTape() as tape:
                loss = self.compute_loss(cloned, x, y)
            grads = tape.gradient(loss, cloned.trainable_variables)
            optim.apply_gradients(zip(grads, cloned.trainable_variables))
        plt.title(f'{name}')
        plt.legend(plots, legends)
        plt.ylim(-5, 5)
        plt.xlim(-6, 6)
        plt.savefig(f'images/{name}.png')
        plt.close()

    def learn_vanilla(self):
        self._learn(self.train_vanilla, 'vanilla')

    def eval_vanilla(self):
        self._evaluate('vanilla')

    def train_maml(self, x, y):
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                loss = self.compute_loss(self.model, x, y)
            grads = tape_inner.gradient(loss, self.model.trainable_variables)
            cloned = self.clone_model()
            k = 0
            for i in range(len(cloned.layers)):  # apply gradient descent to cloned model
                cloned.layers[i].kernel = tf.subtract(self.model.layers[i].kernel, tf.multiply(self.lr_inner, grads[k]))
                cloned.layers[i].bias = tf.subtract(self.model.layers[i].bias, tf.multiply(self.lr_inner, grads[k+1]))
                k += 2
            loss_outer = self.compute_loss(cloned, x, y)
        grads_outer = tape_outer.gradient(loss_outer, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads_outer, self.model.trainable_variables))
        return loss_outer

    def learn_maml(self):
        self._learn(self.train_maml, 'MAML')

    def eval_maml(self):
        self._evaluate('MAML')
