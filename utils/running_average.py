class RunningAverage:

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.metrics = []

    def push(self, metric: float):
        self.metrics.append(metric)
        if len(self.metrics) > self.window_size:
            self.metrics = self.metrics[-self.window_size:]

    def average(self):
        if not self.metrics:
            return 0
        return sum(self.metrics) / len(self.metrics)
