import pytest


@pytest.mark.slow()
def test_foldx_rfp_lambo_runs():
    from poli import create
    import numpy as np

    # For now, we don't have automatic installation of lambo.
    # TODO: add automatic installation of lambo, and remove this
    # check.
    # _ = pytest.importorskip("lambo")

    _, f, _, _, _ = create(name="foldx_rfp_lambo", seed=1)

    # Evaluating on the first base candidate
    first_base_candidate = np.array(
        [
            "AVIKEFMRFKVHMEG"
            "SMNGHEFEIEGEGEGR"
            "PYEGTQTAKLKVTKGG"
            "PLPFSWDILSPQFS"
            "RAFTKHPADIPDYYKQ"
            "SFPEGFKWERVMNFED"
            "GGAVTVTQDTSLED"
            "GTLIYKVKLRGTNFPP"
            "DGPVMQKKTMGWEAST"
            "ERLYPEDGVLKGDI"
            "KMALRLKDGGRYLADF"
            "KTTYKAKKPVQMPGAYN"
            "VDRKLDITSHNEDYTVV"
            "EQYERSEGRHSTG"
        ]
    )
    assert np.isclose(
        f(first_base_candidate), np.array([[-10591.87684184, -61.8757]])
    ).all()


if __name__ == "__main__":
    test_foldx_rfp_lambo_runs()
