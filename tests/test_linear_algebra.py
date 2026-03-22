import numpy as np
import pytest

from matrixbootstrap.linear_algebra import get_real_coefficients_from_dict


def test_get_real_coefficients():
    assert get_real_coefficients_from_dict(x={"A": 0, "B": 1 * 1j, "C": 4 * 1j}) == {
        "A": 0,
        "B": 1,
        "C": 4,
    }
    assert get_real_coefficients_from_dict(x={"A": 0, "B": 1, "C": 4}) == {
        "A": 0,
        "B": 1,
        "C": 4,
    }
    with pytest.raises(ValueError) as exc_info:
        get_real_coefficients_from_dict(x={"A": 0, "B": 1 * 1j, "C": 4})
    assert str(exc_info.value) == "Warning, coefficients are not all real or imaginary."
