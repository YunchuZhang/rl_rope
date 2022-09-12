import numpy as np

class Smoother:
    def __init__(self, dim, window_len, window='hanning'):
        self.window = \
                (np.ones(window_len, 'd') if window == 'flat'
                 else eval('np.'+window+'(window_len)'))
        self.w = np.array(self.window / self.window.sum()).reshape(1, -1)
        self.window_len = window_len
        self.data = np.zeros((window_len, dim))
        self.is_init = False

    def append(self, datapoint):
        if self.is_init:
            self.data[:-1] = self.data[1:]
            self.data[-1,:] = datapoint
        else:
            self.data[:] = datapoint
            self.is_init = True

    def get(self):
        wl = self.window_len // 2
        wr = self.window_len - wl
        return (np.matmul(self.w[:,:wr], self.data[wl:])
              + np.matmul(self.w[:,wr:], self.data[-2:-wl-2:-1])).squeeze()

    def reset(self, new_window_len=None):
        self.is_init = False
        self.window_len = new_window_len or self.window_len

if __name__ == "__main__":
    _s = Smoother(3, 5)
    _s.append([1,2,3])
    assert((_s.get() == [1,2,3]).all())
    _s.append([2,3,4])
    _s.append([3,4,5])
    assert((_s.get() == [2.5,3.5,4.5]).all())

    # TEST SMOOTHING
    import numpy as np
    import matplotlib.pyplot as plt

    def smooth_data(data, window_len=15):
        window_fn = np.hanning(window_len)
        smooth_weights = np.array(window_fn / window_fn.sum()).reshape(1, -1)
        half_window = window_len // 2
        data_len = len(data)
        print(smooth_weights)

        new_data = np.copy(data)
        w_idx = 0
        for _idx in range(half_window, data_len - half_window):
            new_data[_idx] = np.matmul(smooth_weights, data[w_idx:w_idx + window_len, :])
            w_idx += 1

        return new_data

    a = np.arange(100) + np.random.normal(size=100)
    a = a.reshape(20, 5)
    b = smooth_data(a, 5)
    print(np.linalg.norm(a.flatten() - np.arange(100)))
    print(np.linalg.norm(b.flatten() - np.arange(100)))

    plt.plot(np.arange(100), a.flatten(), color='red')
    plt.plot(np.arange(100), b.flatten(), color='blue')
    plt.show()
