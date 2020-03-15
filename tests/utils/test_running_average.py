from utils.running_average import RunningAverage


def test_running_average():
    running_average = RunningAverage(window_size=2)
    running_average.push(metric=1.0)
    running_average.push(metric=2.0)
    assert running_average.average() == 1.5

    running_average.push(metric=3.0)
    assert running_average.average() == 2.5

    running_average.push(metric=-3.0)
    assert running_average.average() == 0


def test_running_average_no_metric():
    running_average = RunningAverage(window_size=2)
    assert running_average.average() == 0


def test_running_average_one_metric():
    running_average = RunningAverage(window_size=2)
    running_average.push(metric=1.0)
    assert running_average.average() == 1.0
