import numpy as np
import pytest

from mkprobes.codebook import codebook
from mkprobes.codebook.codebook import CodebookPicker, CodebookPickerSingleCell


CODES = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0],
    ],
    dtype=bool,
)


def test_single_cell_picker_scores_without_existing_counts() -> None:
    picker = CodebookPickerSingleCell(CODES, ["GeneA", "GeneB"])
    counts = np.array([[1.0, 2.0], [3.0, 4.0]])

    seed, totals = picker.find_optimalish(counts, iterations=1)

    assert seed == 0
    assert totals.shape == (CODES.shape[1],)
    assert np.all(np.isfinite(totals))


def test_single_cell_picker_rejects_zero_expression_total() -> None:
    picker = CodebookPickerSingleCell(CODES, ["GeneA", "GeneB"])

    with pytest.raises(ValueError, match="positive total"):
        picker.find_optimalish(np.zeros((2, 2)), iterations=1)


@pytest.mark.parametrize(
    "fpkm",
    [
        np.zeros(2),
        np.array([1.0, -1.0]),
        np.array([1.0, np.nan]),
    ],
)
def test_bulk_picker_rejects_invalid_expression_totals(fpkm: np.ndarray) -> None:
    picker = CodebookPicker(CODES, ["GeneA", "GeneB"])

    with pytest.raises(ValueError, match="finite and nonnegative|positive total"):
        picker.find_optimalish(fpkm, iterations=1)


@pytest.mark.parametrize("iterations", [0, -1])
def test_picker_rejects_nonpositive_iterations(iterations: int) -> None:
    picker = CodebookPicker(CODES, ["GeneA", "GeneB"])
    single_cell_picker = CodebookPickerSingleCell(CODES, ["GeneA", "GeneB"])

    with pytest.raises(ValueError, match="iterations must be positive"):
        picker.find_optimalish(np.ones(2), iterations=iterations)
    with pytest.raises(ValueError, match="iterations must be positive"):
        single_cell_picker.find_optimalish(np.ones((2, 2)), iterations=iterations)


@pytest.mark.parametrize(
    ("codes", "genes", "message"),
    [
        (CODES[0], ["GeneA"], "two-dimensional"),
        (np.vstack([CODES, CODES[0]]), ["GeneA"], "duplicate codes"),
        (CODES, ["GeneA"] * 5, "Gene names must be unique"),
        (CODES, ["Blanket"], "reserved"),
        (CODES, [f"Gene{i}" for i in range(5)], "more genes than possible codes"),
    ],
)
def test_picker_rejects_invalid_assignment_inputs(
    codes: np.ndarray, genes: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CodebookPicker(codes, genes)


def test_existing_three_bit_code_is_removed_without_reordering_candidates() -> None:
    picker = CodebookPicker(CODES, ["GeneA"], existing=CODES[:1, :3])

    np.testing.assert_array_equal(picker.mhd4, CODES[1:])


def test_capacity_warning_reports_number_of_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(codebook.logger, "warning", messages.append)
    picker = CodebookPicker(CODES, [f"Gene{i}" for i in range(4)])

    picker.find_optimalish(np.ones(4), iterations=1)

    assert "possible codes (4)" in messages[0]
