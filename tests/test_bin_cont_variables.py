import numpy as np
import pytest

from src.data_preprocessing.bin_cont_variables import define_bins, bin_row


def test_define_bins_shapes_and_labels():
    qed_bins, qed_labels, sa_bins, sa_labels = define_bins()

    # QED: 0.0..1.0 step 0.1 => 11 edges, 10 labels
    assert isinstance(qed_bins, np.ndarray)
    assert len(qed_bins) == 11
    assert len(qed_labels) == 10
    assert qed_labels[0] == "0<=qed<0.1"
    assert qed_labels[-1] == "0.9<=qed<1"

    # SA: 1.0..10.0 step 1.0 => 10 edges, 9 labels
    assert isinstance(sa_bins, np.ndarray)
    assert len(sa_bins) == 10
    assert len(sa_labels) == 9
    assert sa_labels[0] == "1<=sa<2"
    assert sa_labels[-1] == "9<=sa<10"


@pytest.mark.parametrize(
    "qed,sa,expected_qed,expected_sa",
    [
        (0.05, 1.1, "0<=qed<0.1", "1<=sa<2"),
        (0.95, 9.9, "0.9<=qed<1", "9<=sa<10"),
        (1.0, 10.0, "0.9<=qed<1", "9<=sa<10"),  # edge values should be clipped in-bin
        (None, 5.0, None, "5<=sa<6"),
        (0.2, None, "0.2<=qed<0.3", None),
        (float("nan"), float("nan"), None, None),
    ],
)
def test_bin_row_assigns_labels(qed, sa, expected_qed, expected_sa):
    qed_bins, qed_labels, sa_bins, sa_labels = define_bins()
    res = bin_row(qed, sa, qed_bins, qed_labels, sa_bins, sa_labels)
    assert res["qed_bin"] == expected_qed
    assert res["sa_bin"] == expected_sa
