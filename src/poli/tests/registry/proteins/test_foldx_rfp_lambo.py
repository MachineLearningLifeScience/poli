import pytest


@pytest.mark.poli__lambo
@pytest.mark.slow()
def test_foldx_rfp_lambo_runs():
    import numpy as np

    from poli import create

    # For now, we don't have automatic installation of lambo.
    # TODO: add automatic installation of lambo, and remove this
    # check.
    problem = create(name="foldx_rfp_lambo", seed=1)
    f = problem.black_box

    # Evaluating on the first base candidate
    first_base_candidate_and_mutation = np.array(
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
            "EQYERSEGRHSTG",
            "IVIKEFMRFKVHMEG"
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
            "EQYERSEGRHSTG",
        ]
    )
    assert np.isclose(
        f(first_base_candidate_and_mutation),
        np.array([[-10591.87684184, -61.8757], [-10634.23150497, -61.5511]]),
    ).all()
