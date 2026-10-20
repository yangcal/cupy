# Cold FFT attribution

Find where a new FFT plan costs CPU time. Every sample calls the public
`cupy.fft` API through the Ajay nvmath adapter with an empty, enabled plan
cache. Plan insertion is inside the call; eviction is timed separately after
the GPU stream has drained. There are no alternative backend comparison rows.

Library initialization, JIT caches and memory pools are warmed. This measures
new-plan requests, not first-process startup. The default preserves the
adapter's workspace-release policy; `--retain-workspace` retains workspace
until eviction instead.

These measurements use the nvmath `execute()` and Workspace3 implementations
from Ajay's candidate branch (`ajay/fft-cache-lifecycle`), pinned to
`84d59bc0fa8128142669ec9586537f5bba169254`, including the inline execution
checks. The CuPy Ajay adapter includes cached dtype names, compiled
byte-to-element stride conversion, common-axis shortcuts and no-op scaling
optimizations.

## Replicate

Build this CuPy branch with the same pinned nvmath candidate using the
[source installation instructions](../fft_nvmath_adapter/README.md#replicate),
checking out `ajay/fft-adapter-cold-path` for CuPy. The nvmath candidate is
`84d59bc0fa8128142669ec9586537f5bba169254`. No profiler package is required.

From the CuPy checkout, check three tiny operands, then run the full sweep:

```sh
python -P benchmarks/fft_nvmath_cold/main.py --suite basic \
  --repeats 1 --json cold-smoke.json
python -P benchmarks/fft_nvmath_cold/main.py --json cold-results.json
```

Use `--list-cases` to inspect coverage and `--case SUBSTRING` to select cases.
Defaults are six balanced blocks, five fresh-plan samples per mode per block.
Use a new output path for each run. CPU-only instrumentation checks:

```sh
python -P benchmarks/fft_nvmath_cold/test_instrumentation.py
```

## Read the results

The three modes run the same public request, in balanced order:

| Mode | What to use it for |
| --- | --- |
| Baseline | Unmodified public-call CPU latency from `cupyx.profiler.benchmark`. |
| Phases | Construction, planning, first execution, axes, key creation, scaling and operand release inside the public call. |
| Detail | Named intervals inside construction, planning, execution and eviction cleanup, with source line numbers. |

Construction includes metadata, CUDA scopes/events, stream resolution,
layout traits, result/workspace allocators, handle creation and workspace
objects. Planning separates preparation, the cuFFT planning call and scope
exit/synchronization. Execution separates stream setup, first workspace/output
allocation, cuFFT submission, scope exit and workspace release. Eviction
includes workspace release, handle destruction and resource cleanup.

| Output | Contents |
| --- | --- |
| `cold-results.json` | Raw samples, validation, summaries, configuration and source hashes. |
| `cold-results.samples.jsonl` | Incremental raw samples, including eviction records. |
| `cold-results.summary.csv` | Public latency, whole-phase times, workspace bytes and diagnostic overhead per case. |
| `cold-results.detail.csv` | Granular intervals and source lines; eviction rows are prefixed `eviction.`. |
| `cold-results.sources/` | Exact diagnostic method implementations with inserted timestamps. |

The 106 curated requests cover 1D/2D/3D sizes, power-of-two and awkward/prime
lengths, C2C/R2C/C2R, forward/inverse directions, norms, single/double
precision, batched/partial/reordered axes and C/F/positive-strided inputs.
Descriptions and generation are in [cases.py](cases.py), extending the
[warm case catalog](../fft_nvmath_adapter/cases.py).

## Results

NVIDIA B200, 2026-10-20, Python 3.12.3 and CUDA 13.4. Six balanced
blocks with five requests per mode gave 30 samples per mode per case.
All 106 cases passed numerical and nvmath-routing checks.

Times below are medians in **CPU µs**. Public is the uninstrumented
`cupyx.profiler.benchmark` result. Init, Plan and Execute come from the
separate phase diagnostic. Execute includes first output/workspace
allocation and workspace release, but excludes the adapter's scaling step.

### Representative requests

| Request / logical transform shape | Public | Init | Plan | Execute |
| --- | ---: | ---: | ---: | ---: |
| C2C `(2,)` | 567.7 | 107.2 | 349.3 | 64.4 |
| C2C `(2,2)` | 660.5 | 107.2 | 436.1 | 70.1 |
| C2C `(2,2,2)` | 731.2 | 108.2 | 499.5 | 73.1 |
| C2C `(1048576,)` | 936.5 | 107.5 | 719.1 | 65.7 |
| R2C `(1048576,)` | 1589.0 | 111.8 | 1358.3 | 75.8 |
| C2R to `(1048576,)` | 1771.4 | 177.7 | 1441.1 | 74.5 |
| R2C `(65537,)` | 5251.1 | 111.5 | 5009.9 | 88.8 |
| C2R to `(65538,)` | 5687.8 | 175.0 | 5341.7 | 86.6 |

### Representative granular attribution

Every interval from `cold.detail.csv` is shown below for three requests.
Names match the CSV's `phase` and `interval` columns. Values are granular
diagnostic medians in CPU µs; `—` means that interval was not executed.
The C2R column gives the logical real output shape. Boundary numbers refer
to the [pinned nvmath `fft.py`](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py); `after` marks a timestamp
inserted after that statement. Each interval runs until the next executed
boundary. The source links require NVIDIA GitLab access, as does the
replication checkout. Exact timestamp placements are also defined in
[instrumentation.py](instrumentation.py).

#### `__init__` — construction

| Interval | Statements covered | Source boundary | C2C `(2,2,2)` | R2C `(65537,)` | C2R to `(1048576,)` |
| --- | --- | --- | ---: | ---: | ---: |
| `entry` | Method dispatch before operand wrapping. | [entry at 826](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L826) | 1.19 | 1.23 | 1.22 |
| `operand_and_options` | `tensor_wrapper.wrap_operand(...)`; `setup_options_and_execution(...)`. | [836](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L836) | 13.77 | 13.97 | 14.04 |
| `types_axes_and_logging` | Rank/dtype checks; result/compute types; logger flags; `validate_axes(...)`. | [841](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L841) | 4.92 | 7.45 | 7.58 |
| `package_device_and_blocking` | `infer_object_package(...)`; select memory space, execution space, device and blocking flags. | [873](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L873) | 5.62 | 5.61 | 5.93 |
| `cuda_scopes_and_event` | Construct `ThreadLocalCudaScopes(...).local`, including its CUDA event; select device scope. | [914](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L914) | 18.67 | 18.43 | 18.77 |
| `device_scope_entry` | Enter `self._device_scope`. | [919](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L919) | 0.75 | 0.75 | 0.76 |
| `stream_resolver` | `create_stream_resolver_for_stateful_api(...)`. | [920](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L920) | 5.31 | 5.31 | 5.30 |
| `stream_resolution` | `self._stream_resolver(stream)` and constructor logging gates. | [928](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L928) | 2.20 | 2.21 | 2.22 |
| `operand_copy_preparation` | Check inplace mode; initialize original-operand key fields; compute original key for C2R. | [939](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L939) | 1.19 | 1.23 | 8.53 |
| `operand_copy` | `_allocate_operand_or_identity(...)`: return input or allocate/copy the C2R internal buffer. | [985](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L985) | 1.01 | 1.05 | 71.30 |
| `operand_flags_and_layout` | Initialize released/stale/C2R-stream flags; construct `StridedLayout(...)`. | [999](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L999) | 4.71 | 4.87 | 4.58 |
| `plan_traits` | `_fft_impl.compute_plan_traits(...)`. | [1032](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1032) | 7.14 | 8.10 | 6.81 |
| `plan_arguments` | Check cached constructor arguments; call `create_xt_plan_args(...)` if needed. | [1046](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1046) | 2.02 | 3.36 | 0.29 |
| `canonical_key` | Call `create_fft_key(...)` if needed; derive output metadata and check logging gates. | [1056](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1056) | 7.26 | 7.42 | 1.25 |
| `result_allocator` | Select `result_class`; `_prepare_result_allocator()`. | [1094](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1094) | 5.66 | 3.30 | 3.12 |
| `workspace_allocator_selection` | Select the supplied allocator or `memory._MEMORY_MANAGER[...]`; choose CUDA handle path. | [1098](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1098) | 1.76 | 1.78 | 1.59 |
| `cuFFT_handle_create` | `cufft.create()`. | [1112](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1112) | 3.41 | 3.55 | 3.29 |
| `cuFFT_handle_configure` | `cufft.set_stream(...)`; `cufft.set_auto_allocation(handle, 0)`. | [after 1112](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1112) | 1.74 | 1.74 | 1.79 |
| `workspace_objects` | Set `fft_planned=False`; enter the CUDA workspace-construction branch. | [1120](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1120) | 0.42 | 0.43 | 0.42 |
| `original_workspace_object` | Construct `Workspace(...)` with allocator, callback and device. | [1130](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1130) | 13.83 | 14.04 | 13.75 |
| `candidate_workspace_object` | Construct `Workspace3(...)`; initialize last event and `valid_state`. | [1139](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1139) | 8.29 | 8.30 | 8.33 |
| `constructor_exit` | Final logging gate and device-scope exit. | [after 1158](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1158) | 0.63 | 0.64 | 0.68 |

#### `plan` — planning

| Interval | Statements covered | Source boundary | C2C `(2,2,2)` | R2C `(65537,)` | C2R to `(1048576,)` |
| --- | --- | --- | ---: | ---: | ---: |
| `entry` | Planning precondition/atomic decorators and method dispatch. | [entry at 1562](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1562) | 3.22 | 3.18 | 3.34 |
| `logging_and_state_checks` | Read logger flags; check `fft_planned`. | [1587](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1587) | 0.69 | 0.69 | 0.71 |
| `device_scope_entry` | Enter device scope; select the CUDA planning branch. | [1595](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1595) | 0.85 | 0.90 | 1.04 |
| `stream_resolution` | `self._stream_resolver(stream)`. | [1601](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1601) | 0.57 | 0.59 | 0.69 |
| `set_stream` | `cufft.set_stream(handle, stream_holder.ptr)`. | [1604](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1604) | 0.57 | 0.58 | 0.59 |
| `callback_options_and_checks` | Resolve prolog/epilog options; `_is_lto_supported_shape(...)`; `set_prolog_and_epilog(...)`; inplace-layout checks. | [1607](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1607) | 2.82 | 2.81 | 2.94 |
| `plan_arguments` | `create_xt_plan_args(...)`; logging gates; select backend. | [1637](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1637) | 1.67 | 2.75 | 4.28 |
| `planning_scope_entry` | Select scopes and enter blocking `scopes.launch_scope(...)`. | [1696](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1696) | 1.64 | 1.62 | 1.64 |
| `cuFFT_make_plan` | `int(cufft.xt_make_plan_many(handle, *plan_args))`. | [1704](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1704) | 477.11 | 4984.90 | 1410.82 |
| `workspace_size_updates` | `set_size(workspace_size)` on active and reference workspace objects. | [after 1704](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1704) | 1.98 | 2.21 | 2.16 |
| `planning_scope_exit_and_sync` | Exit the blocking launch scope, including event recording and synchronization. | [after 1707](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1707) | 7.38 | 8.14 | 8.05 |
| `planning_finalize` | Set `fft_planned=True`; final logging and device-scope exit. | [1709](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L1709) | 2.07 | 2.53 | 2.54 |

#### `execute` — first execution

| Interval | Statements covered | Source boundary | C2C `(2,2,2)` | R2C `(65537,)` | C2R to `(1048576,)` |
| --- | --- | --- | ---: | ---: | ---: |
| `entry` | Check valid/planned state, operand release and C2R staleness. | [entry at 2264](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2264) | 2.02 | 2.41 | 2.35 |
| `device_scope_entry` | Enter `self._device_scope`. | [2306](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2306) | 2.26 | 3.89 | 3.06 |
| `logging_direction_and_allocator_selection` | Read logger flags; select result allocator; resolve/validate direction. | [2307](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2307) | 3.98 | 4.82 | 4.47 |
| `stream_resolution` | `self._stream_resolver(stream)`. | [2317](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2317) | 1.12 | 1.39 | 1.40 |
| `stream_scope_entry` | Select and enter `stream_cm`; CUDA assertions; select scopes. | [2321](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2321) | 4.21 | 4.41 | 4.22 |
| `set_stream` | `cufft.set_stream(handle, stream_holder.ptr)`. | [2359](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2359) | 1.34 | 1.63 | 1.54 |
| `workspace_allocation` | Enter `workspace.allocate_perhaps(...)`; select the out-of-place result branch. | [2365](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2365) | 4.16 | 13.20 | 12.76 |
| `output_allocation` | `allocate_result(stream_holder=stream_holder)`. | [2373](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2373) | 19.38 | 15.33 | 14.49 |
| `execution_launch_preparation` | Read result pointer; logging gates; enter `scopes.launch_scope(...)`. | [after 2373](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2373) | 2.29 | 2.31 | 2.17 |
| `cuFFT_execute` | `cufft.xt_exec(handle, operand_ptr, result_ptr, direction)`. | [2388](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2388) | 15.19 | 27.64 | 17.03 |
| `launch_workspace_scope_exit` | Exit execution launch scope; test `release_workspace`. | [after 2388](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2388) | 1.44 | 1.66 | 1.65 |
| `workspace_release` | `workspace.release(last_compute_event)` and allocation-scope exit. | [2391](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2391) | 14.62 | 7.04 | 6.81 |
| `result_flags_and_return` | Mark C2R operand stale; select result; clear internal result reference; return through scope exits. | [2393](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2393) | 1.88 | 2.94 | 2.97 |

#### `eviction.free` — cleanup outside the public call

| Interval | Statements covered | Source boundary | C2C `(2,2,2)` | R2C `(65537,)` | C2R to `(1048576,)` |
| --- | --- | --- | ---: | ---: | ---: |
| `entry` | Check `valid_state` before cleanup. | [entry at 2555](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2555) | 0.70 | 0.77 | 0.70 |
| `device_scope_entry` | Enter device scope and cleanup block. | [2565](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2565) | 0.81 | 0.85 | 0.90 |
| `operand_ordering` | `_maybe_wait_c2r_aux_buffer_on_last_compute()`. | [2569](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2569) | 0.48 | 0.46 | 0.53 |
| `workspace_release` | Release active/reference workspaces; reset last event; select handle cleanup. | [2570](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2570) | 1.07 | 1.06 | 1.02 |
| `cuFFT_destroy` | `cufft.destroy(handle)`. | [2578](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2578) | 160.80 | 984.88 | 617.72 |
| `handle_finalize` | Clear `self.handle`; finish backend handle-cleanup branch. | [after 2578](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2578) | 0.33 | 0.46 | 0.40 |
| `clear_attributes_and_resources` | Iterate `vars(self)` and clear attributes except logger and valid-state flag. | [2584](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2584) | 12.99 | 14.86 | 14.55 |
| `free_finalize` | Set `valid_state=False`; device-scope exit and final logging gate. | [2594](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2594) | 1.10 | 1.32 | 1.31 |

#### Adapter and operand-release intervals

These phases contain one recorded interval spanning the helper call.
CuPy source links refer to this checkout; other lines refer to the pinned
nvmath `fft.py`.

| Phase / interval | Function covered | Source | C2C `(2,2,2)` | R2C `(65537,)` | C2R to `(1048576,)` |
| --- | --- | --- | ---: | ---: | ---: |
| `axes.entry` | `_normalize_axes_ajay(...)` | [CuPy:94](../../cupy/fft/_fft_nvmath.py#L94) | 3.25 | 1.54 | 1.57 |
| `key.entry` | `_create_key_from_operand(...)` | [CuPy:170](../../cupy/fft/_fft_nvmath.py#L170) | 15.88 | 13.48 | 13.44 |
| `scaling.entry` | `_scale_result_ajay(...)` | [CuPy:148](../../cupy/fft/_fft_nvmath.py#L148) | — | — | 35.14 |
| `release_operand.entry` | `FFT.release_operand()` | [2194](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/blob/84d59bc0fa8128142669ec9586537f5bba169254/nvmath/fft/fft.py#L2194) | 3.61 | 3.95 | 6.66 |


### Full sweep

Operand describes the actual input shape, dtype and layout; C2R inputs are
packed complex spectra. The phase columns have the same meaning as above.
Full scaling, key, axes, release, remainder and observer measurements are
in the generated summary CSV.

| Request | Operand | Public | Init | Plan | Execute |
| --- | --- | ---: | ---: | ---: | ---: |
| `fft(a, axis=-1)` | `(2,)` complex64 C | 567.7 | 107.2 | 349.3 | 64.4 |
| `fft(a, axis=-1)` | `(8,)` complex64 C | 575.0 | 107.0 | 358.6 | 64.3 |
| `fft(a, axis=-1)` | `(64,)` complex64 C | 589.2 | 107.2 | 374.7 | 64.9 |
| `fft(a, axis=-1)` | `(1024,)` complex64 C | 681.9 | 106.2 | 462.1 | 65.6 |
| `fft(a, axis=-1)` | `(65536,)` complex64 C | 810.8 | 107.0 | 597.3 | 64.9 |
| `fft(a, axis=-1)` | `(1048576,)` complex64 C | 936.5 | 107.5 | 719.1 | 65.7 |
| `fftn(a, axes=(0, 1))` | `(2, 2)` complex64 C | 660.5 | 107.2 | 436.1 | 70.1 |
| `fftn(a, axes=(0, 1))` | `(8, 8)` complex64 C | 682.3 | 108.0 | 452.8 | 70.1 |
| `fftn(a, axes=(0, 1))` | `(32, 32)` complex64 C | 708.0 | 108.5 | 479.2 | 71.1 |
| `fftn(a, axes=(0, 1))` | `(128, 128)` complex64 C | 748.0 | 108.1 | 520.9 | 71.7 |
| `fftn(a, axes=(0, 1))` | `(1024, 1024)` complex64 C | 940.6 | 108.7 | 714.5 | 72.4 |
| `fftn(a, axes=(0, 1, 2))` | `(2, 2, 2)` complex64 C | 731.2 | 108.2 | 499.5 | 73.1 |
| `fftn(a, axes=(0, 1, 2))` | `(4, 4, 4)` complex64 C | 747.2 | 107.9 | 514.4 | 73.5 |
| `fftn(a, axes=(0, 1, 2))` | `(16, 16, 16)` complex64 C | 765.7 | 108.1 | 536.5 | 74.0 |
| `fftn(a, axes=(0, 1, 2))` | `(64, 64, 64)` complex64 C | 779.8 | 108.6 | 546.1 | 74.5 |
| `fftn(a, axes=(0, 1, 2))` | `(128, 64, 128)` complex64 C | 834.4 | 108.7 | 604.0 | 75.3 |
| `fft(a, axis=-1, norm='backward')` | `(1024,)` complex64 C | 686.5 | 107.4 | 465.4 | 66.0 |
| `fft(a, axis=-1, norm='forward')` | `(1024,)` complex64 C | 617.5 | 107.6 | 366.8 | 61.7 |
| `fft(a, axis=-1, norm='ortho')` | `(1024,)` complex64 C | 617.7 | 107.3 | 365.6 | 62.4 |
| `ifft(a, axis=-1, norm=None)` | `(1024,)` complex64 C | 617.0 | 107.4 | 366.7 | 61.7 |
| `ifft(a, axis=-1, norm='forward')` | `(1024,)` complex64 C | 584.1 | 106.9 | 366.8 | 61.5 |
| `ifft(a, axis=-1, norm='ortho')` | `(1024,)` complex64 C | 621.0 | 107.3 | 364.7 | 62.4 |
| `ifft(a, axis=-1)` | `(1048576,)` complex64 C | 872.1 | 107.3 | 620.3 | 62.2 |
| `fftn(a, axes=(0, 1), norm='backward')` | `(32, 32)` complex64 C | 601.0 | 107.8 | 374.6 | 66.8 |
| `fftn(a, axes=(0, 1), norm='forward')` | `(32, 32)` complex64 C | 638.9 | 107.0 | 376.0 | 66.5 |
| `fftn(a, axes=(0, 1), norm='ortho')` | `(32, 32)` complex64 C | 639.6 | 106.7 | 378.6 | 67.4 |
| `ifftn(a, axes=(0, 1), norm=None)` | `(32, 32)` complex64 C | 641.2 | 107.5 | 373.6 | 66.8 |
| `ifftn(a, axes=(0, 1), norm='forward')` | `(32, 32)` complex64 C | 598.9 | 108.1 | 379.6 | 66.9 |
| `ifftn(a, axes=(0, 1), norm='ortho')` | `(32, 32)` complex64 C | 637.8 | 107.3 | 375.8 | 67.8 |
| `ifftn(a, axes=(0, 1))` | `(1024, 1024)` complex64 C | 872.8 | 107.2 | 609.6 | 69.1 |
| `fftn(a, axes=(0, 1, 2), norm='backward')` | `(16, 16, 16)` complex64 C | 662.8 | 107.5 | 435.8 | 70.7 |
| `fftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16)` complex64 C | 702.7 | 108.2 | 434.1 | 70.5 |
| `fftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16)` complex64 C | 704.3 | 107.4 | 433.1 | 71.6 |
| `ifftn(a, axes=(0, 1, 2), norm=None)` | `(16, 16, 16)` complex64 C | 701.1 | 107.5 | 433.6 | 70.5 |
| `ifftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16)` complex64 C | 662.0 | 107.7 | 434.2 | 70.8 |
| `ifftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16)` complex64 C | 701.5 | 107.9 | 431.2 | 70.5 |
| `ifftn(a, axes=(0, 1, 2))` | `(128, 64, 128)` complex64 C | 770.9 | 108.2 | 502.4 | 71.3 |
| `fftn(a, axes=(1,))` | `(3, 16)` complex64 C | 486.4 | 108.7 | 266.6 | 60.5 |
| `fftn(a, axes=(1,))` | `(16, 65536)` complex64 C | 713.6 | 109.0 | 492.7 | 61.7 |
| `fft2(a)` | `(2, 8, 16)` complex64 C | 584.3 | 108.5 | 354.9 | 67.4 |
| `fft2(a)` | `(4, 256, 512)` complex64 C | 773.4 | 108.1 | 544.7 | 69.2 |
| `fftn(a, axes=(1, 0))` | `(8, 16)` complex64 C | 585.5 | 108.5 | 356.3 | 67.6 |
| `fftn(a, axes=(1, 0))` | `(1024, 1024)` complex64 C | 837.6 | 108.8 | 610.7 | 69.2 |
| `fftn(a, axes=(0, 2))` | `(4, 3, 8)` complex64 C | 585.8 | 110.7 | 352.6 | 68.6 |
| `fftn(a, axes=(0, 2))` | `(128, 4, 1024)` complex64 C | 715.0 | 110.6 | 477.9 | 70.7 |
| `fftn(a, axes=[0, 2])` | `(4, 3, 8)` complex64 C | 582.4 | 110.1 | 352.5 | 68.1 |
| `fftn(a, axes=[0, 2])` | `(128, 4, 1024)` complex64 C | 717.1 | 111.0 | 483.8 | 70.9 |
| `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(4, 3, 8)` complex64 C | 585.2 | 109.8 | 354.8 | 68.5 |
| `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(128, 4, 1024)` complex64 C | 723.2 | 111.2 | 484.9 | 71.0 |
| `fftn(a, axes=(0, -1))` | `(8, 16)` complex64 C | 586.0 | 107.7 | 357.7 | 67.3 |
| `fftn(a, axes=(0, -1))` | `(1024, 1024)` complex64 C | 841.5 | 108.6 | 612.5 | 69.8 |
| `fftn(a, axes=(2, 3))` | `(2, 3, 8, 16)` complex64 C | 588.3 | 108.6 | 357.6 | 67.9 |
| `fftn(a, axes=(2, 3))` | `(2, 4, 256, 512)` complex64 C | 775.6 | 108.7 | 548.8 | 69.8 |
| `ifft2(a, norm=None)` | `(2, 8, 16)` complex64 C | 626.3 | 108.0 | 355.9 | 66.9 |
| `ifft2(a, norm='forward')` | `(2, 8, 16)` complex64 C | 586.0 | 107.6 | 359.7 | 67.5 |
| `ifft2(a, norm='ortho')` | `(2, 8, 16)` complex64 C | 628.1 | 108.2 | 357.5 | 68.1 |
| `ifft2(a, norm=None)` | `(4, 256, 512)` complex64 C | 817.8 | 109.2 | 549.9 | 70.1 |
| `ifft2(a, norm='forward')` | `(4, 256, 512)` complex64 C | 776.2 | 109.2 | 548.0 | 69.3 |
| `ifft2(a, norm='ortho')` | `(4, 256, 512)` complex64 C | 816.9 | 108.6 | 546.5 | 70.0 |
| `rfft(a)` | `(32,)` float32 C | 464.4 | 111.7 | 249.4 | 59.8 |
| `rfft(a)` | `(1048576,)` float32 C | 1589.0 | 111.8 | 1358.3 | 75.8 |
| `irfft(a)` | `(17,)` complex64 C | 554.4 | 172.4 | 243.9 | 53.6 |
| `irfft(a)` | `(524289,)` complex64 C | 1771.4 | 177.7 | 1441.1 | 74.5 |
| `rfft2(a)` | `(8, 16)` float32 C | 528.9 | 111.4 | 302.7 | 65.2 |
| `rfft2(a)` | `(1024, 1024)` float32 C | 721.6 | 112.6 | 493.5 | 68.5 |
| `irfft2(a)` | `(8, 9)` complex64 C | 629.5 | 174.5 | 304.6 | 60.0 |
| `irfft2(a)` | `(1024, 513)` complex64 C | 827.1 | 174.8 | 491.8 | 65.1 |
| `rfftn(a)` | `(4, 8, 16)` float32 C | 603.6 | 111.8 | 372.2 | 68.4 |
| `rfftn(a)` | `(128, 64, 128)` float32 C | 657.5 | 112.6 | 423.8 | 70.5 |
| `irfftn(a)` | `(4, 8, 9)` complex64 C | 710.5 | 178.1 | 376.4 | 64.8 |
| `irfftn(a)` | `(128, 64, 65)` complex64 C | 762.3 | 178.6 | 426.7 | 66.3 |
| `rfft(a, norm='forward')` | `(33,)` float32 C | 492.2 | 111.1 | 241.5 | 59.9 |
| `rfft(a, norm='ortho')` | `(1048575,)` float32 C | 2023.7 | 110.8 | 1746.0 | 84.6 |
| `irfft2(a, norm='forward')` | `(8, 9)` complex64 C | 591.5 | 173.4 | 302.1 | 60.1 |
| `irfft2(a, norm='ortho')` | `(512, 513)` complex64 C | 827.7 | 174.3 | 496.7 | 65.9 |
| `fft(a, norm=None)` | `(1048576,)` complex128 C | 885.8 | 107.8 | 672.5 | 63.1 |
| `ifft(a, norm=None)` | `(65536,)` complex128 C | 800.1 | 107.3 | 550.3 | 64.0 |
| `rfftn(a, norm='forward')` | `(64, 64, 64)` float64 C | 678.7 | 112.5 | 404.8 | 69.4 |
| `irfftn(a, norm='ortho')` | `(64, 64, 33)` complex128 C | 746.4 | 177.8 | 408.4 | 64.6 |
| `fft(a)` | `(8, 16)` complex64 F | 449.9 | 110.3 | 231.4 | 59.9 |
| `fft(a)` | `(16, 65536)` complex64 F | 715.6 | 110.2 | 497.4 | 61.8 |
| `fft(a)` | `(64,)` complex64 strided | 444.8 | 106.8 | 232.9 | 59.1 |
| `fft(a)` | `(1048576,)` complex64 strided | 836.7 | 107.4 | 628.6 | 62.5 |
| `ifft(a)` | `(8, 16)` complex64 F | 485.4 | 109.3 | 230.7 | 59.8 |
| `ifft(a)` | `(16, 65536)` complex64 F | 754.3 | 110.1 | 499.7 | 61.4 |
| `ifft(a)` | `(64,)` complex64 strided | 482.7 | 107.5 | 234.0 | 59.3 |
| `ifft(a)` | `(1048576,)` complex64 strided | 876.7 | 108.7 | 627.7 | 62.6 |
| `fft(a)` | `(127,)` complex64 C | 455.2 | 106.8 | 242.5 | 60.4 |
| `rfft(a)` | `(127,)` float32 C | 445.5 | 111.4 | 230.7 | 59.5 |
| `fft(a)` | `(1009,)` complex64 C | 1158.5 | 107.6 | 944.6 | 64.7 |
| `rfft(a)` | `(1009,)` float32 C | 2696.5 | 110.7 | 2471.1 | 72.0 |
| `fft(a)` | `(65537,)` complex64 C | 2793.9 | 107.5 | 2559.0 | 80.5 |
| `rfft(a)` | `(65537,)` float32 C | 5251.1 | 111.5 | 5009.9 | 88.8 |
| `fftn(a)` | `(31, 33)` complex64 C | 588.5 | 108.5 | 362.0 | 68.5 |
| `rfftn(a)` | `(31, 33)` float32 C | 555.7 | 112.5 | 325.1 | 66.5 |
| `fftn(a)` | `(127, 129)` complex64 C | 765.3 | 109.2 | 543.3 | 67.7 |
| `rfftn(a)` | `(127, 129)` float32 C | 1207.5 | 112.2 | 969.4 | 78.7 |
| `fftn(a)` | `(7, 9, 11)` complex64 C | 627.2 | 109.1 | 399.9 | 70.5 |
| `rfftn(a)` | `(7, 9, 11)` float32 C | 626.1 | 113.0 | 396.9 | 69.8 |
| `fftn(a)` | `(31, 33, 35)` complex64 C | 645.7 | 108.9 | 415.9 | 70.4 |
| `rfftn(a)` | `(31, 33, 35)` float32 C | 639.3 | 112.9 | 409.6 | 70.4 |
| `irfft(a)` | `(64,)` complex64 C | 553.6 | 172.6 | 240.6 | 54.0 |
| `irfft(a)` | `(505,)` complex64 C | 570.6 | 175.5 | 256.8 | 54.4 |
| `irfft(a)` | `(32770,)` complex64 C | 5687.8 | 175.0 | 5341.7 | 86.6 |
| `irfftn(a)` | `(30, 18)` complex64 C | 1026.0 | 174.8 | 694.5 | 71.4 |
| `irfftn(a)` | `(6, 10, 8)` complex64 C | 735.5 | 176.8 | 400.5 | 65.8 |
