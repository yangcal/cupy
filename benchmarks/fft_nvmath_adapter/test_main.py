"""GPU-free checks for the targeted C2R benchmark comparison."""

from __future__ import annotations

from collections import Counter
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PATH = Path(__file__).with_name("main.py")
SPEC = importlib.util.spec_from_file_location("_fft_driver_test", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class TestC2RComparison(unittest.TestCase):
    def test_targeted_catalog_can_be_listed_without_gpu_imports(self):
        output = io.StringIO()
        with patch.object(sys, "argv", [str(PATH), "--case", "irfft",
                                       "--compare-c2r-reset", "--list-cases"]):
            with contextlib.redirect_stdout(output):
                driver.main()
        self.assertEqual(
            output.getvalue().count("original/fused/ajay/native"), 9)
        self.assertTrue(output.getvalue().endswith("9 cases\n"))

    def test_targeted_mode_rejects_mixed_cases_and_unbalanced_blocks(self):
        for arguments in (["--compare-c2r-reset"],
                          ["--compare-c2r-reset", "--case", "irfft"]):
            with self.subTest(arguments=arguments):
                with patch.object(sys, "argv", [str(PATH), *arguments]):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            driver.main()
                self.assertEqual(error.exception.code, 2)

    def test_four_row_orders_balance_positions_and_adjacent_pairs(self):
        labels = ("original", "fused", "ajay", "native")
        orders = driver.row_orders(labels, 24)
        self.assertEqual(len(set(orders)), 24)
        self.assertEqual(orders, driver.row_orders(labels, 24))
        for index in range(4):
            self.assertEqual(Counter(order[index] for order in orders),
                             Counter({label: 6 for label in labels}))
        adjacent = Counter(pair for order in orders
                           for pair in zip(order, order[1:]))
        self.assertEqual(len(adjacent), 12)
        self.assertEqual(set(adjacent.values()), {6})

    def test_preparation_regression_is_not_hidden_by_original_savings(self):
        values = {"original": 60., "fused": 40., "ajay": 45., "native": 50.}
        rows = {label: {"cpu_median_us": value,
                        "gpu_median_us": 2 * value}
                for label, value in values.items()}
        pairs = {name: (a, b) for name, (a, b) in driver.PAIRS.items()
                 if a in values and b in values}
        block = {"rows": rows, "paired_cpu_us": {
            name: values[a] - values[b] for name, (a, b) in pairs.items()},
            "paired_gpu_us": {
            name: 2 * (values[a] - values[b])
            for name, (a, b) in pairs.items()}}
        result = driver.summarize([block] * 24, tuple(values))
        self.assertFalse(result["negative_median"])
        self.assertFalse(result["material_regression_candidate"])
        self.assertTrue(result["fused_negative_median"])
        self.assertTrue(result["fused_material_regression_candidate"])
        self.assertEqual(result["fused_minus_ajay"], -5.)
        self.assertEqual(result["paired_gpu_us"]["fused_minus_ajay"], -10.)


if __name__ == "__main__":
    unittest.main()
