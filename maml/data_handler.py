import numpy as np


class Sinusoid:
    def __init__(self, K=10, amplitude=None, phase=None):
        self.K = K
        self.amplitude = amplitude if amplitude else np.random.uniform(0.1, 5.0)
        self.phase = phase if phase else np.random.uniform(0, np.pi)
        self.sampled_points = None
        self.x = self._sample_x()

    def _sample_x(self):
        return np.random.uniform(-5, 5, self.K)

    def f(self, x):
        return self.amplitude*np.sin(x - self.phase)

    def batch(self, x=None, force_new=None):
        if x is None:
            x = self._sample_x() if force_new else self.x
        y = self.f(x)
        return x[:, None].astype('float32'), y[:, None].astype('float32')

    def equally_spaced_samples(self, K=None):
        if K is None:
            K = self.K
        return self.batch(x=np.linspace(-5, 5, K))


def data_gen(K, train_size=20000, test_size=10):
    def _data_gen(size):
        return [Sinusoid(K) for _ in range(size)]
    return _data_gen(train_size), _data_gen(test_size)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot(data, *args, **kwargs):
        x, y = data
        return plt.plot(x, y, *args, **kwargs)

    plt.title('Sinusoid examples')
    for _ in range(3):
        plot(Sinusoid(K=100).equally_spaced_samples())
    plt.savefig('sinusoid_example.png')
    plt.close()
