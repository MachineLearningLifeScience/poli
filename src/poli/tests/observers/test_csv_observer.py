import csv

import numpy as np
import pytest

from poli.core.exceptions import ObserverNotInitializedError
from poli.core.util.observers.csv_observer import CSVObserver, CSVObserverInitInfo
from poli.repository import AlohaBlackBox


def test_csv_observer_logs_on_aloha():
    f = AlohaBlackBox()
    observer = CSVObserver()
    observer.initialize_observer(
        f.info,
        {
            "experiment_id": "test_csv_observer_logs_on_aloha",
            "experiment_path": "./poli_results",
        },
        seed=0,
    )

    f.set_observer(observer)
    f(np.array([list("MIGUE")]))
    f(np.array([list("ALOOF")]))
    f(np.array([list("ALOHA"), list("OMAHA")]))

    assert observer.csv_file_path.exists()

    # Loading up the csv and checking results
    with open(observer.csv_file_path, "r") as f:
        reader = csv.reader(f)
        results = list(reader)

    assert results[0] == ["x", "y"]
    assert results[1][0] == "MIGUE" and float(results[1][1]) == 0.0
    assert results[2][0] == "ALOOF" and float(results[2][1]) == 3.0
    assert results[3][0] == "ALOHA" and float(results[3][1]) == 5.0
    assert results[4][0] == "OMAHA" and float(results[4][1]) == 2.0


def test_csv_observer_works_with_incomplete_caller_info():
    f = AlohaBlackBox()
    observer = CSVObserver()
    observer.initialize_observer(
        f.info,
        {},
        seed=0,
    )

    f.set_observer(observer)
    f(np.array([list("MIGUE")]))
    f(np.array([list("ALOOF")]))
    f(np.array([list("ALOHA"), list("OMAHA")]))

    assert observer.csv_file_path.exists()

    # Loading up the csv and checking results
    with open(observer.csv_file_path, "r") as f:
        reader = csv.reader(f)
        results = list(reader)

    assert results[0] == ["x", "y"]
    assert results[1][0] == "MIGUE" and float(results[1][1]) == 0.0
    assert results[2][0] == "ALOOF" and float(results[2][1]) == 3.0
    assert results[3][0] == "ALOHA" and float(results[3][1]) == 5.0
    assert results[4][0] == "OMAHA" and float(results[4][1]) == 2.0


def test_observer_without_initialization():
    f = AlohaBlackBox()
    observer = CSVObserver()

    f.set_observer(observer)

    with pytest.raises(ObserverNotInitializedError):
        f(np.array([list("MIGUE")]))


def test_works_with_csv_init_object():
    f = AlohaBlackBox()
    observer = CSVObserver()
    observer.initialize_observer(
        f.info,
        CSVObserverInitInfo(
            experiment_id="test_csv_observer_logs_on_aloha",
            experiment_path="./poli_results",
        ),
        seed=0,
    )
    f.set_observer(observer)
    f(np.array([list("MIGUE")]))
    f(np.array([list("ALOOF")]))
    f(np.array([list("ALOHA"), list("OMAHA")]))
    assert observer.csv_file_path.exists()
    # Loading up the csv and checking results
    with open(observer.csv_file_path, "r") as f:
        reader = csv.reader(f)
        results = list(reader)
    assert results[0] == ["x", "y"]
    assert results[1][0] == "MIGUE" and float(results[1][1]) == 0.0
    assert results[2][0] == "ALOOF" and float(results[2][1]) == 3.0
    assert results[3][0] == "ALOHA" and float(results[3][1]) == 5.0
    assert results[4][0] == "OMAHA" and float(results[4][1]) == 2.0
