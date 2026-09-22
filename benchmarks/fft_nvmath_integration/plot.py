# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: MIT

"""Reduce and plot native/nvmath public FFT integration timings."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics


COLORS = {
    "native": "#4C78A8",
    "nvmath": "#E45756",
}
METRICS = (
    ("host", "host-return"),
    ("completion", "synchronized completion"),
    ("event", "CUDA-event span"),
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot reduce an empty sample list")
    position = (len(ordered) - 1) * percentile / 100
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _corrected(samples: list[int], clock_pairs: list[int]) -> list[float]:
    if len(samples) != len(clock_pairs):
        raise ValueError("sample and clock-pair counts differ")
    return [
        max(0, sample - clock) / 1000
        for sample, clock in zip(samples, clock_pairs)
    ]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "p90_us": _percentile(values, 90),
        "p99_us": _percentile(values, 99),
    }


def _reduce(payload: dict) -> dict[str, dict]:
    reduced = {}
    for name, case in payload["cases"].items():
        host = _corrected(case["host_samples_ns"], case["host_clock_pairs_ns"])
        completion = _corrected(
            case["completion_samples_ns"],
            case["completion_clock_pairs_ns"],
        )
        event = [sample / 1000 for sample in case["event_samples_ns"]]
        reduced[name] = {
            "mode": case["mode"],
            "fixture": case["fixture"],
            "backend": case["backend"],
            "host": _stats(host),
            "completion": _stats(completion),
            "event": _stats(event),
            "host_clock_pair_mean_ns": statistics.fmean(
                case["host_clock_pairs_ns"]
            ),
            "completion_clock_pair_mean_ns": statistics.fmean(
                case["completion_clock_pairs_ns"]
            ),
        }
    return reduced


def _case(
    reduced: dict[str, dict],
    mode: str,
    fixture: str,
    backend: str,
) -> dict | None:
    return reduced.get(f"{mode}:{fixture}:{backend}")


def _print_report(payload: dict, reduced: dict[str, dict]) -> None:
    device = payload["environment"]["device_name"]
    print(f"Device: {device}")
    print()
    for mode in payload["mode_order"]:
        print(f"## {mode} public-call latency")
        print()
        print(
            f"{'fixture':34s} {'backend':8s} "
            f"{'host mean':>12s} {'host p90':>12s} "
            f"{'sync mean':>12s} {'event span':>12s}"
        )
        print("-" * 97)
        for fixture in payload["fixture_order"]:
            for backend in payload["backend_order"]:
                row = _case(reduced, mode, fixture, backend)
                if row is None:
                    continue
                print(
                    f"{fixture:34s} {backend:8s} "
                    f"{row['host']['mean_us']:11.3f} us "
                    f"{row['host']['p90_us']:11.3f} us "
                    f"{row['completion']['mean_us']:11.3f} us "
                    f"{row['event']['mean_us']:11.3f} us"
                )

            native = _case(reduced, mode, fixture, "native")
            nvmath = _case(reduced, mode, fixture, "nvmath")
            if native is None or nvmath is None:
                continue
            host_gap = nvmath["host"]["mean_us"] - native["host"]["mean_us"]
            completion_gap = (
                nvmath["completion"]["mean_us"]
                - native["completion"]["mean_us"]
            )
            print(
                f"  gap (nvmath-native): host={host_gap:+.3f} us; "
                f"synchronized={completion_gap:+.3f} us"
            )
        print()

    if "cold" in payload["mode_order"]:
        candidates = []
        for fixture in payload["fixture_order"]:
            native = _case(reduced, "cold", fixture, "native")
            nvmath = _case(reduced, "cold", fixture, "nvmath")
            if native is None or nvmath is None:
                continue
            native_host = native["host"]["mean_us"]
            nvmath_host = nvmath["host"]["mean_us"]
            candidates.append(
                (native_host, fixture, nvmath_host - native_host)
            )
        if candidates:
            print("## Cold cases ranked by native host-return latency")
            print()
            print(f"{'fixture':34s} {'native':>12s} {'nvmath-native':>16s}")
            print("-" * 66)
            for native_host, fixture, gap in sorted(candidates):
                print(f"{fixture:34s} {native_host:11.3f} us {gap:+15.3f} us")
            print()


def _paired_plot(
    payload: dict,
    reduced: dict[str, dict],
    mode: str,
    output: pathlib.Path,
) -> None:
    import matplotlib.pyplot as plt

    fixtures = [
        fixture
        for fixture in payload["fixture_order"]
        if any(
            _case(reduced, mode, fixture, backend) is not None
            for backend in payload["backend_order"]
        )
    ]
    if not fixtures:
        return

    height = max(5.5, 0.36 * len(fixtures) + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(18, height), sharey=True)
    positions = list(range(len(fixtures)))
    offsets = {"native": -0.10, "nvmath": 0.10}

    for axis, (metric, title) in zip(axes, METRICS):
        for backend in payload["backend_order"]:
            xs = []
            ys = []
            for position, fixture in zip(positions, fixtures):
                row = _case(reduced, mode, fixture, backend)
                if row is None:
                    continue
                xs.append(row[metric]["mean_us"])
                ys.append(position + offsets.get(backend, 0))
            axis.scatter(
                xs,
                ys,
                label=backend,
                color=COLORS[backend],
                s=28,
                zorder=3,
            )

        if set(payload["backend_order"]) >= {"native", "nvmath"}:
            for position, fixture in zip(positions, fixtures):
                native = _case(reduced, mode, fixture, "native")
                nvmath = _case(reduced, mode, fixture, "nvmath")
                if native is None or nvmath is None:
                    continue
                axis.plot(
                    [native[metric]["mean_us"], nvmath[metric]["mean_us"]],
                    [position - 0.10, position + 0.10],
                    color="#B5B5B5",
                    linewidth=1,
                    zorder=1,
                )

        axis.set_title(title)
        axis.set_xlabel("corrected mean time (microseconds)")
        axis.grid(axis="x", linestyle=":", alpha=0.45)
        if mode == "cold":
            axis.set_xscale("log")

    axes[0].set_yticks(positions, labels=fixtures)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, loc="best")
    device = payload["environment"]["device_name"]
    fig.suptitle(
        f"{mode.capitalize()} public CuPy FFT: native vs nvmath on {device}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _candidate_plot(
    payload: dict,
    reduced: dict[str, dict],
    output: pathlib.Path,
) -> None:
    import matplotlib.pyplot as plt

    points = []
    for fixture in payload["fixture_order"]:
        native = _case(reduced, "cold", fixture, "native")
        nvmath = _case(reduced, "cold", fixture, "nvmath")
        if native is None or nvmath is None:
            continue
        baseline = native["host"]["mean_us"]
        gap = nvmath["host"]["mean_us"] - baseline
        points.append((fixture, baseline, gap))
    if not points:
        return

    fig, axis = plt.subplots(figsize=(12, 8))
    axis.axhline(0, color="#777777", linewidth=1)
    axis.scatter(
        [point[1] for point in points],
        [point[2] for point in points],
        color="#E45756",
        s=36,
        zorder=3,
    )
    for fixture, baseline, gap in points:
        axis.annotate(
            fixture,
            (baseline, gap),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_xlabel("native CuPy cold host-return mean (microseconds)")
    axis.set_ylabel("nvmath - native cold host-return mean (microseconds)")
    axis.set_title(
        "Cold FFT candidates: cheap native planning vs integration gap"
    )
    axis.grid(linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = json.loads(args.json.read_text())
    reduced = _reduce(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _print_report(payload, reduced)
    for mode in payload["mode_order"]:
        output = args.output_dir / f"{mode}.png"
        _paired_plot(payload, reduced, mode, output)
        if output.exists():
            print(f"{mode.capitalize()} plot: {output}")
    if "cold" in payload["mode_order"]:
        output = args.output_dir / "cold-candidates.png"
        _candidate_plot(payload, reduced, output)
        if output.exists():
            print(f"Cold-candidate plot: {output}")


if __name__ == "__main__":
    main()
