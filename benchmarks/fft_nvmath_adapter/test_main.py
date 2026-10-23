"""GPU-free checks for warm FFT benchmark row selection and measurement."""

from __future__ import annotations

import ast
from collections import Counter
from itertools import product
import hashlib
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).with_name("main.py")
SPEC = importlib.util.spec_from_file_location("_fft_driver_test", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class TestC2RComparison(unittest.TestCase):
    def test_default_rows_balance_positions_and_adjacent_pairs(self):
        for minimal in (False, True):
            with self.subTest(minimal=minimal):
                labels = driver.row_labels({'minimal': minimal})
                expected = ('original', 'axes', 'fused', 'c2r', 'ajay')
                expected += (('minimal', 'native', 'stateful') if minimal
                             else ('native',))
                self.assertEqual(labels, expected)
                size = len(labels)
                orders = driver.row_orders(labels, 24)
                self.assertEqual(len(orders), 24)
                self.assertEqual(orders, driver.row_orders(labels, 24))
                self.assertEqual(len(set(orders)), size)
                for index in range(size):
                    self.assertEqual(
                        Counter(order[index] for order in orders),
                        Counter({label: 24 // size for label in labels}))
                adjacent = Counter(pair for order in orders
                                   for pair in zip(order, order[1:]))
                self.assertEqual(
                    set(adjacent),
                    {(a, b) for a in labels for b in labels if a != b})
                self.assertEqual(set(adjacent.values()), {24 // size})

    def test_default_rows_select_matching_adapter_and_reset(self):
        tree = ast.parse(PATH.read_text())
        hooks = next(node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name)
                             and target.id == 'hooks'
                             for target in node.targets))
        select = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == 'select')
        original, unfused, fused = object(), object(), object()
        current, reference = object(), object()
        cached = type('CachedFFT', (), {})
        adapter = SimpleNamespace(
            CachedFFT=cached, _try_use_nvmath=original,
            _try_use_nvmath_ajay_unfused=unfused,
            _try_use_nvmath_ajay=fused)
        namespace = {
            'adapter': adapter, 'nvmath_reset': current,
            'reset_reference': SimpleNamespace(
                reset_operand_unchecked=reference),
            'cache': (), 'select_reset': driver.select_reset,
            'cp': SimpleNamespace(fft=SimpleNamespace(
                config=SimpleNamespace(use_nvmath=None))),
            'fft_module': SimpleNamespace(),
        }
        namespace['hooks'] = eval(
            compile(ast.Expression(hooks), str(PATH), 'eval'), namespace)
        exec(compile(ast.Module(body=[select], type_ignores=[]), str(PATH),
                     'exec'), namespace)
        expected = {
            'original': (original, reference),
            'axes': (unfused, reference),
            'fused': (fused, reference),
            'c2r': (unfused, current),
            'ajay': (fused, current),
            'native': (original, current),
        }
        for label, (hook, reset) in expected.items():
            with self.subTest(row=label):
                namespace['select'](label)
                self.assertIs(namespace['fft_module']._try_use_nvmath, hook)
                self.assertIs(cached.reset_operand_unchecked, reset)
                self.assertEqual(
                    namespace['cp'].fft.config.use_nvmath, label != 'native')

    def test_reference_reset_snapshot_is_pinned(self):
        path = PATH.with_name("reference_reset.py")
        source = path.read_text()
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef))
        method = "\n".join(source.splitlines()[
            function.lineno - 1:function.end_lineno]) + "\n"
        self.assertEqual(hashlib.sha256(method.encode()).hexdigest(),
                         driver.REFERENCE_RESET_SHA)
        self.assertNotIn("_try_reset_c2r_operand", method)
        self.assertNotIn("_reset_operand_unchecked_prepared", method)

    def test_reset_selection_changes_live_plans_but_not_nvmath(self):
        class FFT:
            def execute(self):
                return 'execute'

            def reset_operand_unchecked(self, operand, *, stream=None):
                return ('prepared', operand, stream)

        class CachedFFT(FFT):
            pass

        def reference_reset(self, operand, *, stream=None):
            return ('reference', operand, stream)

        prepared_reset = FFT.reset_operand_unchecked
        execute = FFT.execute
        plan = CachedFFT()
        for reset, expected in ((reference_reset, 'reference'),
                                (prepared_reset, 'prepared'),
                                (reference_reset, 'reference')):
            driver.select_reset(CachedFFT, reset)
            self.assertEqual(plan.reset_operand_unchecked('input', stream=7),
                             (expected, 'input', 7))
            self.assertIs(type(plan), CachedFFT)
            self.assertIs(CachedFFT.reset_operand_unchecked, reset)
            self.assertIs(CachedFFT.execute, execute)
            self.assertIs(FFT.reset_operand_unchecked, prepared_reset)
        self.assertEqual(FFT().reset_operand_unchecked(None),
                         ('prepared', None, None))

    def test_reset_selection_clears_only_c2r_usage_diagnostics(self):
        class CachedFFT:
            pass

        def reset(self, operand, *, stream=None):
            pass

        c2r = SimpleNamespace(fft_abstract_type='C2R',
                              _c2r_reset_prepared_last_used=True,
                              _c2r_reset_signature=('cached',))
        c2c = SimpleNamespace(fft_abstract_type='C2C')
        driver.select_reset(CachedFFT, reset, (c2r, c2c))
        self.assertFalse(c2r._c2r_reset_prepared_last_used)
        self.assertEqual(c2r._c2r_reset_signature, ('cached',))
        self.assertFalse(hasattr(c2c, '_c2r_reset_prepared_last_used'))
        self.assertIs(CachedFFT.reset_operand_unchecked, reset)

    def test_finally_restores_inherited_or_owned_reset_after_failure(self):
        main = next(node for node in ast.parse(PATH.read_text()).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'main')
        outer = next(node for node in main.body
                     if isinstance(node, ast.Try) and node.finalbody)
        restore = next(node for node in outer.finalbody
                       if isinstance(node, ast.If)
                       and isinstance(node.test, ast.Compare)
                       and isinstance(node.test.left, ast.Name)
                       and node.test.left.id == 'previous_reset')
        cleanup = compile(ast.Module(body=[restore], type_ignores=[]),
                          str(PATH), 'exec')

        def inherited(self, operand, *, stream=None):
            pass

        def owned(self, operand, *, stream=None):
            pass

        def temporary(self, operand, *, stream=None):
            pass

        for existing, selected in product((None, owned), (False, True)):
            with self.subTest(previously_owned=existing is not None,
                              selected=selected):
                base = type('FFT', (), {'reset_operand_unchecked': inherited})
                cached = type('CachedFFT', (base,), {})
                if existing is not None:
                    cached.reset_operand_unchecked = existing
                missing = object()
                previous = vars(cached).get('reset_operand_unchecked', missing)
                namespace = {'adapter': SimpleNamespace(CachedFFT=cached),
                             'previous_reset': previous, 'missing': missing}
                if selected:
                    driver.select_reset(cached, temporary)
                with self.assertRaisesRegex(RuntimeError, 'benchmark failed'):
                    try:
                        raise RuntimeError('benchmark failed')
                    finally:
                        exec(cleanup, namespace)
                self.assertIs(cached.reset_operand_unchecked,
                              inherited if existing is None else owned)
                self.assertEqual('reset_operand_unchecked' in vars(cached),
                                 existing is not None)
                self.assertIs(base.reset_operand_unchecked, inherited)

    def test_shared_plan_diagnostics_follow_current_rows_reset(self):
        function = next(node for node in ast.walk(ast.parse(PATH.read_text()))
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'verify_c2r_reset')

        def prepared_reset(self, operand, *, stream=None):
            self._c2r_reset_prepared_last_used = True

        def reference_reset(self, operand, *, stream=None):
            pass

        cached = type('CachedFFT', (), {
            'reset_operand_unchecked': prepared_reset})
        plan = cached()
        plan.fft_abstract_type = 'C2R'
        plan._c2r_reset_prepared_info = {'eligible': True}
        plan._c2r_reset_signature = ('recipe from preceding Ajay row',)
        observations = {}
        namespace = {
            'check': driver.check,
            'adapter': SimpleNamespace(CachedFFT=cached),
            'original_plan': plan, 'fused_plan': plan, 'plan': plan,
            'separate_scale_plan': plan,
            'nvmath_reset': prepared_reset, 'reset_observations': observations,
            'reset_reference': SimpleNamespace(
                reset_operand_unchecked=reference_reset)}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(PATH), 'exec'), namespace)
        verify = namespace['verify_c2r_reset']
        for label in ('original', 'axes', 'fused'):
            driver.select_reset(cached, reference_reset, (plan,))
            plan.reset_operand_unchecked(None)
            verify(label)
            self.assertEqual(observations[label]['c2r_preparation'], {
                'selected': False, 'info': None, 'last_used': False})
        driver.select_reset(cached, prepared_reset, (plan,))
        with self.assertRaisesRegex(RuntimeError, 'was not used'):
            verify('ajay')
        plan.reset_operand_unchecked(None)
        with self.assertRaisesRegex(RuntimeError, 'Reference control used'):
            verify('fused')
        verify('ajay')
        verify('c2r')
        self.assertEqual(observations['ajay']['c2r_preparation'], {
            'selected': True, 'info': {'eligible': True}, 'last_used': True})
        self.assertEqual(observations['c2r']['c2r_preparation'],
                         observations['ajay']['c2r_preparation'])
        plan._c2r_reset_prepared_info['eligible'] = False
        driver.select_reset(cached, reference_reset, (plan,))
        plan._c2r_reset_prepared_last_used = True
        with self.assertRaisesRegex(RuntimeError, 'Reference control used'):
            verify('fused')
        plan._c2r_reset_prepared_last_used = False
        verify('fused')
        self.assertTrue(observations['ajay']['c2r_preparation']['last_used'])
        self.assertTrue(observations['ajay']['c2r_preparation']['info'][
            'eligible'])
        self.assertFalse(observations['fused']['c2r_preparation']['last_used'])
        self.assertEqual(plan._c2r_reset_signature,
                         ('recipe from preceding Ajay row',))

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
                          ["--compare-c2r-reset", "--case", "irfft",
                           "--blocks", "30"]):
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
