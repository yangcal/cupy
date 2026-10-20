"""Diagnostic-only boundary timers for the pinned nvmath candidate.

Methods are cloned from installed source with timestamp calls inserted before
or after named statements. Original operations, branches and cleanup remain.
No tracing, line_profiler, or edits to the installed nvmath source are used.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import textwrap
from time import perf_counter_ns

FFT_SOURCE_SHA = (
    "12d56bcbe5f6c6a55389cf6f2e01f2e52f347b536ae181418758bbb01dbbf089"
)

# (interval starting here, statement prefix, before/after)
BOUNDARIES = {
    "__init__": [
        (
            "operand_and_options",
            "self.operand = operand = tensor_wrapper.wrap_operand",
            False,
        ),
        ("types_axes_and_logging", "self.operand_dim =", False),
        ("package_device_and_blocking", "self.package =", False),
        ("cuda_scopes_and_event", "self._cuda_scopes =", False),
        ("device_scope_entry", "with self._device_scope:", False),
        ("stream_resolver", "self._stream_resolver =", False),
        (
            "stream_resolution",
            "stream_holder = self._stream_resolver(stream)",
            False,
        ),
        ("operand_copy_preparation", "self.inplace =", False),
        ("operand_copy", "self.operand, self.operand_backup =", False),
        ("operand_flags_and_layout", "self._operand_released =", False),
        ("plan_traits", "self.plan_traits =", False),
        (
            "plan_arguments",
            "if self.plan_args_from_user_operand is None:",
            False,
        ),
        ("canonical_key", "if self.key_from_user_operand is None:", False),
        ("result_allocator", "self.result_class =", False),
        (
            "workspace_allocator_selection",
            "if self.execution_space == 'cpu':\n    self.allocator = None",
            False,
        ),
        ("cuFFT_handle_create", "self.handle = cufft.create()", False),
        ("cuFFT_handle_configure", "self.handle = cufft.create()", True),
        ("workspace_objects", "self.fft_planned =", False),
        (
            "original_workspace_object",
            "self._workspace_original: Workspace | None =",
            False,
        ),
        (
            "candidate_workspace_object",
            "self.workspace: Workspace | Workspace3 | None =",
            False,
        ),
        ("constructor_exit", "self.valid_state = True", True),
    ],
    "plan": [
        ("logging_and_state_checks", "log_info =", False),
        ("device_scope_entry", "with self._device_scope:", False),
        (
            "stream_resolution",
            "stream_holder = self._stream_resolver(stream)",
            False,
        ),
        ("set_stream", "cufft.set_stream(", False),
        (
            "callback_options_and_checks",
            "prolog = utils.check_or_create_options",
            False,
        ),
        ("plan_arguments", "plan_args = create_xt_plan_args(", False),
        ("planning_scope_entry", "scopes = self._cuda_scopes", False),
        (
            "cuFFT_make_plan",
            "workspace_size = int(cufft.xt_make_plan_many(",
            False,
        ),
        (
            "workspace_size_updates",
            "workspace_size = int(cufft.xt_make_plan_many(",
            True,
        ),
        # Following size updates: launch-scope event record and sync.
        (
            "planning_scope_exit_and_sync",
            "self._workspace_original.set_size(workspace_size)",
            True,
        ),
        ("planning_finalize", "self.fft_planned = True", False),
    ],
    "execute": [
        ("device_scope_entry", "with self._device_scope:", False),
        ("logging_direction_and_allocator_selection", "log_info =", False),
        (
            "stream_resolution",
            "stream_holder = self._stream_resolver(stream)",
            False,
        ),
        ("stream_scope_entry", "stream_cm =", False),
        ("set_stream", "cufft.set_stream(", False),
        (
            "workspace_allocation",
            "with self.workspace.allocate_perhaps(",
            False,
        ),
        (
            "output_allocation",
            "self.result = allocate_result(stream_holder=stream_holder)",
            False,
        ),
        (
            "execution_launch_preparation",
            "self.result = allocate_result(stream_holder=stream_holder)",
            True,
        ),
        ("cuFFT_execute", "cufft.xt_exec(", False),
        ("launch_workspace_scope_exit", "cufft.xt_exec(", True),
        (
            "workspace_release",
            "self.workspace.release(self.last_compute_event)",
            False,
        ),
        (
            "result_flags_and_return",
            "if self.fft_abstract_type == 'C2R':",
            False,
        ),
    ],
    "free": [
        ("device_scope_entry", "with self._device_scope:", False),
        (
            "operand_ordering",
            "self._maybe_wait_c2r_aux_buffer_on_last_compute()",
            False,
        ),
        ("workspace_release", "if self.workspace is not None:", False),
        ("cuFFT_destroy", "cufft.destroy(self.handle)", False),
        ("handle_finalize", "cufft.destroy(self.handle)", True),
        ("clear_attributes_and_resources", "_keep =", False),
        ("free_finalize", "self.valid_state = False", False),
    ],
}


class Recorder:
    """Sequential intervals, including observer overhead; no subtraction."""

    def __init__(self):
        self.active = None
        self.events = []

    def mark(self, phase, name, line):
        if self.active == phase:
            timestamp = perf_counter_ns()
            self.events.append((name, line, timestamp))

    def begin(self, phase):
        if self.active is not None:
            raise RuntimeError("Nested diagnostic phases are unsupported")
        self.events = []
        self.active = phase
        self.mark(phase, "entry", None)

    def finish(self):
        end = perf_counter_ns()
        events = self.events
        self.active = None
        return [
            {
                "name": name,
                "line": line,
                "ns": (events[i + 1][2] if i + 1 < len(events) else end)
                - timestamp,
            }
            for i, (name, line, timestamp) in enumerate(events)
        ]


def clone(method, phase, recorder, owner=None):
    method = inspect.unwrap(method)
    source_lines, first_line = inspect.getsourcelines(method)
    tree = ast.parse(textwrap.dedent("".join(source_lines)))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise RuntimeError(f"{phase}: expected a Python method")
    if ast.get_docstring(function, clean=False) is not None:
        function.body[0].value.value = method.__doc__
    statements = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.stmt) and not isinstance(n, ast.FunctionDef)
    ]
    insertions = {}
    anchors = []
    for name, prefix, after in BOUNDARIES[phase]:
        matches = [n for n in statements if ast.unparse(n).startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"{phase}/{name}: expected one anchor, "
                f"found {len(matches)}: {prefix}"
            )
        node = matches[0]
        line = first_line + node.lineno - 1
        call = ast.Expr(
            ast.Call(
                ast.Name("_cold_boundary", ast.Load()),
                [ast.Constant(phase), ast.Constant(name), ast.Constant(line)],
                [],
            )
        )
        insertions.setdefault(id(node), []).append((after, call))
        anchors.append(
            {"name": name, "line": line, "after": after, "statement": prefix}
        )

    class Insert(ast.NodeTransformer):
        def visit(self, node):
            original_id = id(node)
            result = super().visit(node)
            if original_id not in insertions:
                return result
            before = [
                call for after, call in insertions[original_id] if not after
            ]
            after = [call for after, call in insertions[original_id] if after]
            return before + [result] + after

    tree = Insert().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(method.__globals__)
    if owner is not None:
        namespace.update(vars(owner))
    namespace["_cold_boundary"] = recorder.mark
    filename = inspect.getsourcefile(method)
    exec(compile(tree, f"<cold diagnostic: {phase}>", "exec"), namespace)
    return namespace[function.name], {
        "original_file": filename,
        "anchors": anchors,
        "instrumented_source": ast.unparse(tree),
    }


class Diagnostics:
    """Temporary phase wrappers and optional source-boundary timers."""

    def __init__(self, fft_class, adapter):
        self.recorder = Recorder()
        self.records = []
        self.targets = [(fft_class, name, name) for name in BOUNDARIES]
        self.targets += [(fft_class, "release_operand", "release_operand")]
        self.targets += [
            (adapter, name, phase)
            for name, phase in (
                ("_normalize_axes_ajay", "axes"),
                ("_create_key_from_operand", "key"),
                ("_scale_result_ajay", "scaling"),
            )
        ]
        self.originals = {
            (id(owner), name): getattr(owner, name)
            for owner, name, _ in self.targets
        }
        self.clones = {
            name: clone(
                getattr(fft_class, name), name, self.recorder, fft_class
            )
            for name in BOUNDARIES
        }
        self.source_map = {
            name: metadata for name, (_, metadata) in self.clones.items()
        }

    def wrap(self, function, phase):
        def measured(*args, **kwargs):
            start = perf_counter_ns()
            self.recorder.begin(phase)
            try:
                return function(*args, **kwargs)
            finally:
                spans = self.recorder.finish()
                elapsed = perf_counter_ns() - start
                self.records.append(
                    {"phase": phase, "ns": elapsed, "spans": spans}
                )

        return measured

    @contextlib.contextmanager
    def activate(self, mode):
        try:
            if mode != "baseline":
                for owner, name, phase in self.targets:
                    function = self.originals[(id(owner), name)]
                    if mode == "detail" and name in self.clones:
                        function = self.clones[name][0]
                    setattr(owner, name, self.wrap(function, phase))
            self.records = []
            yield
        finally:
            for owner, name, _ in self.targets:
                setattr(owner, name, self.originals[(id(owner), name)])
            self.recorder.active = None

    def take(self):
        records, self.records = self.records, []
        return records
