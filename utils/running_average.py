class RunningAverage:

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.metrics = []

    def push(self, metric: float):
        self.metrics.append(metric)

    metric = input(20)
    n = int(metric)
    average = 0
    sum = 0
    for num in range(0, n + 1, 1):
        sum = sum + num
    average = sum / n


def average(self):
    pass
