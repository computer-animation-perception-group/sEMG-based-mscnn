from nose.tools import assert_equal
from numpy.testing import assert_array_equal
import numpy as np
from .. import Dataset, Combo


def test_get_trial():
    dataset = Dataset.from_name('ninapro-db1-raw/semg-glove')
    get_trial = dataset.get_trial_func()

    def do(subject, gesture, trial):
        trial = get_trial(Combo(subject=subject, gesture=gesture, trial=trial))
        assert_array_equal(trial.subject, subject)
        assert_equal(trial.gesture[0], -1)
        assert_equal(trial.gesture[-1], gesture)
        assert np.in1d(trial.gesture, [-1, gesture]).all()

    for subject, gesture, trial in [(1, 1, 1),
                                    (27, 1, 1),
                                    (1, 1, 10),
                                    (1, 12, 1),
                                    (1, 13, 1),
                                    (1, 29, 1),
                                    (1, 30, 1),
                                    (1, 52, 1)]:
        yield do, subject, gesture, trial


def test_get_trial_norest():
    dataset = Dataset.from_name('ninapro-db1-raw/semg-glove')
    get_trial = dataset.get_trial_func(norest=True)

    def do(subject, gesture, trial):
        trial = get_trial(Combo(subject=subject, gesture=gesture, trial=trial))
        assert_array_equal(trial.subject, subject)
        assert_array_equal(trial.gesture, gesture)

    for subject, gesture, trial in [(1, 1, 1),
                                    (27, 1, 1),
                                    (1, 1, 10),
                                    (1, 12, 1),
                                    (1, 13, 1),
                                    (1, 29, 1),
                                    (1, 30, 1),
                                    (1, 52, 1)]:
        yield do, subject, gesture, trial
