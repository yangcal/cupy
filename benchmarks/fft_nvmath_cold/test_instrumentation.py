"""CPU-only checks for diagnostic semantics, restoration and aggregation."""

from __future__ import annotations

import contextlib
import functools
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "cold_instrumentation_tested", HERE / "instrumentation.py"
)
profiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiling)


def checked(predicate):
    def decorate(function):
        @functools.wraps(function)
        def guarded(self, *args, **kwargs):
            predicate(self)
            try:
                return function(self, *args, **kwargs)
            finally:
                self.events.append("decorator_cleanup")

        return guarded

    return decorate


class Example:
    def __init__(self):
        self.events = []
        self.value = 0

    def _allowed(self):
        self.events.append("precondition")

    @contextlib.contextmanager
    def scope(self):
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")

    @checked(_allowed)
    def plan(self, *, fail=False):
        """Original method with decorators, a scope, branch and return."""
        self.events.append("body")
        with self.scope():
            if fail:
                raise ValueError("expected failure")
            self.value += 2
        return self.value

    def release_operand(self):
        return None


RECIPE = {
    "plan": [
        ("body", "self.events.append('body')", False),
        ("scope", "with self.scope():", False),
        ("increment", "self.value += 2", False),
    ]
}


class Tests(unittest.TestCase):
    def test_clone_preserves_operations_and_decorators(self):
        recorder = profiling.Recorder()
        with patch.dict(profiling.BOUNDARIES, RECIPE, clear=True):
            clone, meta = profiling.clone(
                Example.plan, "plan", recorder, Example
            )
        for fail in (False, True):
            original, diagnostic = Example(), Example()
            if fail:
                with self.assertRaisesRegex(ValueError, "expected failure"):
                    original.plan(fail=True)
            else:
                expected = original.plan()
            recorder.begin("plan")
            try:
                if fail:
                    with self.assertRaisesRegex(
                        ValueError, "expected failure"
                    ):
                        clone(diagnostic, fail=True)
                else:
                    self.assertEqual(clone(diagnostic), expected)
            finally:
                spans = recorder.finish()
            self.assertEqual(original.events, diagnostic.events)
            self.assertEqual(original.value, diagnostic.value)
            self.assertTrue(all(s["ns"] >= 0 for s in spans))
        self.assertEqual(len(meta["anchors"]), 3)
        self.assertIn("@checked(_allowed)", meta["instrumented_source"])

    def test_restore_after_failure(self):
        adapter = types.SimpleNamespace(
            _normalize_axes_ajay=lambda *a: None,
            _create_key_from_operand=lambda *a: None,
            _scale_result_ajay=lambda *a: None,
        )
        original = Example.plan
        with patch.dict(profiling.BOUNDARIES, RECIPE, clear=True):
            diagnostics = profiling.Diagnostics(Example, adapter)
        with self.assertRaisesRegex(ValueError, "expected failure"):
            with diagnostics.activate("detail"):
                Example().plan(fail=True)
        self.assertIs(Example.plan, original)
        self.assertIsNone(diagnostics.recorder.active)
        self.assertEqual(diagnostics.take()[0]["phase"], "plan")

    def test_ambiguous_anchor_is_rejected(self):
        with patch.dict(
            profiling.BOUNDARIES,
            {"plan": [("ambiguous", "self.", False)]},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "expected one anchor"):
                profiling.clone(
                    Example.plan, "plan", profiling.Recorder(), Example
                )

    def test_baseline_is_unmodified(self):
        adapter = types.SimpleNamespace(
            _normalize_axes_ajay=lambda *a: None,
            _create_key_from_operand=lambda *a: None,
            _scale_result_ajay=lambda *a: None,
        )
        with patch.dict(profiling.BOUNDARIES, RECIPE, clear=True):
            diagnostics = profiling.Diagnostics(Example, adapter)
        original = Example.plan
        with diagnostics.activate("baseline"):
            self.assertIs(Example.plan, original)
            Example().plan()
        self.assertEqual(diagnostics.take(), [])


if __name__ == "__main__":
    unittest.main()
