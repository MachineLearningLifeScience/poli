from poli.repository import ToyContinuousBlackBox


def test_dynamic_info_on_toy_continuous_black_box():
    f = ToyContinuousBlackBox(
        function_name="ackley_function_01",
        n_dimensions=4,
    )

    assert f.info.max_sequence_length == 4
