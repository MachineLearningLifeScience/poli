from pathlib import Path

import numpy as np

from poli import objective_factory

from mlflow_observer import MlFlowObserver

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    TRACKING_URI = THIS_DIR / "mlruns"
    observer = MlFlowObserver(tracking_uri=TRACKING_URI)

    alphabet = ["", "[C]", "..."]

    problem_info, f, x0, y0, run_info = objective_factory.create(
        name="rdkit_logp",
        observer=observer,
        alphabet=alphabet,
        string_representation="SELFIES",
    )

    f(np.array([["[C]", "[C]"]]))
    f(np.array([["[C]", "[C]", "[C]"]]))
    f(np.array([["[C]", "[C]", "[C]", "[C]"]]))
    
    observer.finish()
