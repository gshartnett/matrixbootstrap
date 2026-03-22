import pytest

from matrixbootstrap.linear_algebra import get_real_coefficients_from_dict


def test_get_real_coefficients_from_imaginary_dict():
    result = get_real_coefficients_from_dict(x={"A": 0, "B": 1j, "C": 4j})
    assert result == {"A": 0, "B": 1, "C": 4}


def test_get_real_coefficients_from_real_dict():
    result = get_real_coefficients_from_dict(x={"A": 0, "B": 1, "C": 4})
    assert result == {"A": 0, "B": 1, "C": 4}


def test_get_real_coefficients_raises_on_mixed():
    with pytest.raises(ValueError):
        get_real_coefficients_from_dict(x={"A": 0, "B": 1j, "C": 4})
