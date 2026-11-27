"""Symbols and atomic numbers for whole arrays (alframework.tools.elements)."""
import numpy as np
import pytest

from alframework.tools.elements import atomic_numbers_of, symbol_text

SYMBOLS = [['H', 'C', 'O', 'Cl'], ['Zn', 'F', 'H', 'H']]
NUMBERS = [[1, 6, 8, 17], [30, 9, 1, 1]]


@pytest.mark.parametrize('values', [np.array(SYMBOLS), np.char.encode(np.array(SYMBOLS), 'utf-8'),
                                    np.array(SYMBOLS, dtype=object), SYMBOLS])
def test_symbols_in_any_text_form_give_atomic_numbers(values):
    found = atomic_numbers_of(values)
    assert found.dtype == np.int64
    np.testing.assert_array_equal(found, NUMBERS)


def test_atomic_numbers_pass_through_and_unknown_symbols_raise():
    numbers = np.array(NUMBERS, dtype=np.int32)
    assert atomic_numbers_of(numbers) is numbers
    with pytest.raises(KeyError):
        atomic_numbers_of(np.array([b'Xx']))
    with pytest.raises(KeyError):
        atomic_numbers_of(['Qq'])
    with pytest.raises(ValueError, match='chemical symbols or atomic numbers'):
        atomic_numbers_of(np.zeros(3))


def test_symbol_text_decodes_stored_bytes():
    assert [symbol_text(value) for value in (b'Cl', np.bytes_(b'H'), 'O')] == ['Cl', 'H', 'O']
