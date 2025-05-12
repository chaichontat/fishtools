from pathlib import Path
from typing import TYPE_CHECKING

import loguru
import numpy as np
import o_codebook as cb
import pytest
from loguru import logger

if TYPE_CHECKING:
    from loguru import Message, Record


@pytest.fixture
def capture_loguru_logs(monkeypatch: pytest.MonkeyPatch):
    """Fixture to capture loguru logs into a list."""
    captured_logs: list["loguru.Record"] = []

    def TmpSink(message: "loguru.Message"):
        captured_logs.append(message.record)

    log_id = logger.add(TmpSink)
    yield captured_logs
    logger.remove(log_id)


class TestGenCodebook:
    @pytest.fixture
    def temp_static_dir_fixture(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """
        Creates a temporary directory to act as 'static', patches cb.static,
        and ensures the directory exists.
        """
        # tmp_path is already a Path object to a unique temporary directory
        monkeypatch.setattr(cb, "static", tmp_path)
        # Ensure the directory exists (though tmp_path itself is a dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        return tmp_path

    @pytest.fixture
    def populated_static_dir(self, temp_static_dir_fixture: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """
        Populates the temporary static directory with dummy codebook CSVs.
        Also patches cb.ns to reflect these dummy files.
        """
        static_dir = temp_static_dir_fixture

        # Create dummy CSV files that CodebookPicker can read.
        # Content: comma-separated bits, one code per line.
        # These codes must have 3 'on' bits as expected by CodebookPicker's typical usage
        # in this context (though CodebookPicker itself might be more general).

        # 10-bit codes, 3 'on'
        content_10bit = (
            "1,1,1,0,0,0,0,0,0,0\n"
            "1,1,0,1,0,0,0,0,0,0\n"
            "1,0,1,1,0,0,0,0,0,0\n"
            "0,1,1,1,0,0,0,0,0,0\n"
            "1,0,0,0,1,1,0,0,0,0"  # 5 codes
        )
        (static_dir / "10bit_on3_dist2.csv").write_text(content_10bit)

        # 12-bit codes, 3 'on'
        content_12bit = (
            "1,1,1,0,0,0,0,0,0,0,0,0\n"
            "0,0,0,1,1,1,0,0,0,0,0,0\n"
            "0,0,0,0,0,0,1,1,1,0,0,0"  # 3 codes
        )
        (static_dir / "12bit_on3_dist2.csv").write_text(content_12bit)

        # Patch cb.ns to reflect exactly these files for predictable test behavior.
        # The original ns population logic scans `cb.static`.
        monkeypatch.setattr(
            cb,
            "ns",
            {
                10: 5,  # 5 codes in 10bit_on3_dist2.csv
                12: 3,  # 3 codes in 12bit_on3_dist2.csv
            },
        )
        return static_dir

    def test_gen_codebook_selects_smallest_sufficient_codebook(
        self, populated_static_dir: Path, capture_loguru_logs: list["loguru.Record"]
    ):
        """
        Tests if gen_codebook selects the smallest bit-size codebook that can accommodate the genes.
        Relies on the patched cb.ns.
        """
        tss = ["GeneA", "GeneB"]  # Needs 2 codes. 10-bit (cap 5) is sufficient.

        # We expect 10-bit to be chosen.
        # cb.static is already patched by populated_static_dir -> temp_static_dir_fixture
        codebook = cb.gen_codebook(tss=tss, offset=0, n_bits=None, seed=0)

        assert "GeneA" in codebook
        assert "GeneB" in codebook
        assert len(codebook) >= len(tss)

        for gene_codes in codebook.values():
            assert len(gene_codes) == 3, "Each gene should be assigned 3 bit positions"
            for bit_pos in gene_codes:
                assert isinstance(bit_pos, int), "Bit positions should be integers"

        expected_log_message = "Using 10-bit codebook with capacity 5."
        found_log = any(expected_log_message in record["message"] for record in capture_loguru_logs)
        assert found_log, f"Expected log message '{expected_log_message}' not found."

    def test_gen_codebook_raises_error_if_no_suitable_codebook(self, populated_static_dir: Path):
        """Tests ValueError when no codebook in ns can accommodate the number of genes."""
        tss = ["G1", "G2", "G3", "G4", "G5", "G6"]  # Needs 6 codes, max capacity in patched ns is 5.

        with pytest.raises(ValueError, match=f"No suitable codebook found. {len(tss)} genes found."):
            cb.gen_codebook(tss=tss, n_bits=None, seed=0)

    def test_gen_codebook_with_offset(
        self, populated_static_dir: Path, capture_loguru_logs: list["loguru.Record"]
    ):
        """
        Tests if offset correctly shifts the bit numbers from the 'order' list.
        This test is conceptual for 'offset' as its direct effect depends on 'order'
        and CodebookPicker's output. We check that it runs and uses the chosen codebook.
        """
        tss = ["GeneA"]
        offset_val = 10

        codebook = cb.gen_codebook(tss=tss, offset=offset_val, n_bits=10, seed=0)

        assert "GeneA" in codebook
        all_bits_used = set()
        for gene_code_list in codebook.values():
            for bit_val in gene_code_list:
                all_bits_used.add(bit_val)

        expected_log_message = "Using 10-bit codebook with capacity 5."
        found_log = any(expected_log_message in record["message"] for record in capture_loguru_logs)
        assert found_log, f"Expected log message '{expected_log_message}' not found."

    def test_gen_codebook_forbidden_swap_logic(
        self,
        populated_static_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capture_loguru_logs: list["loguru.Record"],
    ):
        test_order_list = list(range(100, 200))  # Fill with non-problematic numbers
        test_order_list[0] = 1
        test_order_list[1] = 9
        test_order_list[2] = 17
        monkeypatch.setattr(cb, "order", tuple(test_order_list))

        tss = ["GeneA", "GeneB"]

        codebook = cb.gen_codebook(tss=tss, offset=0, n_bits=10, seed=0)

        assert "GeneA" in codebook
        assert "GeneB" in codebook

        if "Blank-1" in codebook:
            assert codebook["Blank-1"] == [1, 9, 17]
            assert codebook["GeneA"] != [1, 9, 17], (
                "GeneA's original forbidden code should have been swapped out"
            )
        else:
            if codebook["GeneA"] == [1, 9, 17]:
                pytest.fail("GeneA was assigned a forbidden code [1,9,17] but no swap occurred.")

        expected_log_message = "Using 10-bit codebook with capacity 5."
        found_log = any(expected_log_message in record["message"] for record in capture_loguru_logs)
        assert found_log, f"Expected log message '{expected_log_message}' not found."
