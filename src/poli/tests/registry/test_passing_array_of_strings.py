"""This module tests whether giving black boxes an array of b strings
is equivalent to giving them an array of [b, L] tokens."""

from typing import List

import pytest


# TODO: parametrize by all non-aligned blackboxes
@pytest.mark.parametrize(
    "black_box_name, example_non_flat_input, example_flat_input, kwargs",
    [
        [
            "aloha",
            [["A", "L", "O", "O", "F"], ["A", "L", "O", "H", "A"]],
            [
                "ALOOF",
                "ALOHA",
            ],
            {},
        ],
        [
            "dockstring",
            [
                ["C", ""],
                ["C", "C"],
            ],
            [
                "C",
                "CC",
            ],
            {
                "target_name": "ABL1",
            },
        ],
        # We remove drd3 docking from this check, because
        # it doesn't register out-of-the-box. It needs the user
        # to download a couple of files before using it.
        # [
        #     "drd3_docking",
        #     [
        #         ["C", ""],
        #         ["C", "C"],
        #     ],
        #     [
        #         "C",
        #         "CC",
        #     ],
        #     {},
        # ],
        # TODO: add foldx-related black boxes
        # TODO: Once we can add lambo automatically, we can
        # uncomment this test:
        # [
        #     "penalized_logp_lambo",
        #     [
        #         ["C", ""],
        #         ["C", "C"],
        #     ],
        #     [
        #         "C",
        #         "CC",
        #     ],
        #     {},
        # ],
        # TODO: add rasp
        [
            "rdkit_logp",
            [
                ["C", ""],
                ["C", "C"],
            ],
            [
                "C",
                "CC",
            ],
            {},
        ],
        [
            "rdkit_qed",
            [
                ["C", ""],
                ["C", "C"],
            ],
            [
                "C",
                "CC",
            ],
            {},
        ],
        [
            "sa_tdc",
            [
                ["C", ""],
                ["C", "C"],
            ],
            [
                "C",
                "CC",
            ],
            {},
        ],
    ],
)
def test_passing_array_of_strings(
    black_box_name: str,
    example_non_flat_input: List[List[str]],
    example_flat_input: List[str],
    kwargs: dict,
):
    """This test checks whether passing an array of strings [b,]
    to a black box is equivalent to passing an array of [b, L] tokens.
    """
    import numpy as np

    from poli import create

    problem = create(
        name=black_box_name,
        **kwargs,
    )
    f = problem.black_box

    x_flat = np.array(example_flat_input)
    x_non_flat = np.array(example_non_flat_input)

    assert np.array_equal(f(x_flat), f(x_non_flat), equal_nan=True)
