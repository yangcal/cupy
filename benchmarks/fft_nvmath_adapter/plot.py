"""Plot the warm comparison in the same operand order as report.py.

Requires Matplotlib only for rendering; the benchmark itself does not need it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from report import grouped_rows


SERIES = (
    ('original', 'Original', '#e87924', '-', 1.9),
    ('axes', 'Axes', '#a88d16', '--', 1.2),
    ('fused', 'Fusion', '#955fa4', '--', 1.2),
    ('c2r', 'C2R', '#299eaa', '--', 1.2),
    ('ajay', 'Ajay', '#176db0', '-', 2.2),
    ('native', 'Native', '#414141', '-', 1.9),
    ('minimal', 'Minimal', '#3c9859', ':', 1.5),
    ('stateful', 'Stateful', '#9d75cf', ':', 1.5),
)

GROUP_LABELS = (
    'C2R input preservation\nand normalization',
    'C2C normalization\ncontiguous operands',
    'R2C normalization',
    'Batches and axes variants',
    'C2C normalization\nFortran / strided operands',
)


def plot(payload, destination):
    """Render recorded summaries, without recomputing benchmark statistics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if payload['status'] != 'complete':
        raise ValueError('Only a complete benchmark can be plotted')
    groups = grouped_rows(payload['cases'])
    if not groups:
        raise ValueError('No measured cases to plot')
    cases = [case for _, rows in groups for case in rows]
    x = np.arange(len(cases))
    env, config = payload['environment'], payload['configuration']

    with plt.rc_context({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, (latency, delta) = plt.subplots(
            2, 1, figsize=(28, 12), sharex=True,
            gridspec_kw={'height_ratios': (3.2, 1.35), 'hspace': 0.12})
        fig.subplots_adjust(left=0.06, right=0.995, top=0.80, bottom=0.16)

        gpu = env.get('gpu', 'GPU unspecified')
        cpu = env.get('cpu_model', 'CPU unspecified')
        node = env.get('node', env.get('host', {}).get('node', 'unknown node'))
        arch = env.get('host', {}).get('machine', '')
        python = env.get('python', 'unknown').split()[0]
        fig.suptitle(f'Warm FFT CPU latency · {gpu}', y=0.974,
                     fontsize=21)
        fig.text(0.5, 0.941,
                 f'{cpu} ({arch}) · {node} · Python {python}',
                 ha='center', fontsize=13)
        fig.text(0.5, 0.908,
                 f"{len(cases)} cases · {config['blocks']} balanced blocks × "
                 f"{config['repeats']} samples · warmed plans · CPU host time",
                 ha='center', color='#555555')

        offset = 0
        for index, (name, rows) in enumerate(groups):
            end = offset + len(rows)
            color = '#eaf2f8' if index == 0 else (
                '#f2f3f5' if index % 2 == 1 else '#ffffff')
            for axis in (latency, delta):
                axis.axvspan(offset - 0.5, end - 0.5,
                             facecolor=color, zorder=0)
                if offset:
                    axis.axvline(offset - 0.5, color='#ccd0d5',
                                 linewidth=0.7, zorder=1)
            label = GROUP_LABELS[index] if len(groups) == 5 else name
            latency.text((offset + end - 1) / 2, 1.035, label,
                         transform=latency.get_xaxis_transform(),
                         ha='center', va='bottom', fontsize=10)
            offset = end

        for row, label, color, style, width in SERIES:
            values = [case['summary']['rows_cpu_us'].get(row, np.nan)
                      for case in cases]
            if not np.all(np.isnan(values)):
                latency.plot(x, values, label=label, color=color,
                             linestyle=style, linewidth=width, marker='o',
                             markersize=2.4, zorder=4 if row == 'ajay' else 3)
        latency.set_ylabel('CPU call time (µs)', fontsize=13)
        latency.set_ylim(bottom=0)
        latency.margins(y=0.12)
        latency.legend(loc='upper left', ncol=8, frameon=False,
                       fontsize=10, handlelength=2.7, columnspacing=1.5)

        gaps = [case['summary']['ajay_minus_native'] for case in cases]
        delta.bar(x, gaps, width=0.78,
                  color=['#4d9a70' if value <= 0 else '#b65d5d'
                         for value in gaps], zorder=3)
        # These summaries are medians of paired block differences, not
        # differences of the independently summarized row medians above.
        delta.axhline(0, color='#676767', linewidth=1.1, zorder=4)
        delta.set_ylabel('Paired Ajay − Native\nCPU time (µs)', fontsize=12)
        delta.set_xlabel(
            'Case ID (same operand ordering as the README; original catalog '
            'IDs retained)', labelpad=12, fontsize=12)
        delta.set_xticks(x, [str(case['case_id']) for case in cases],
                         rotation=90, fontsize=8)
        delta.set_xlim(-0.7, len(cases) - 0.3)
        for axis in (latency, delta):
            axis.set_axisbelow(True)
            axis.grid(axis='y', color='#dce0e5', linewidth=0.65)
            axis.spines['left'].set_color('#888888')
            axis.spines['bottom'].set_color('#888888')

        fig.text(0.06, 0.06,
                 'Absolute values: median of block medians. '
                 'Gaps: median of paired block differences. '
                 'Negative gap = Ajay faster.', color='#555555', fontsize=10)
        fig.text(0.06, 0.035,
                 'Axes: axes/no-op scaling shortcuts. Fusion: Axes + NVRTC '
                 'runtime scaling. C2R: Axes + prepared copying. '
                 'Ajay: both. Minimal/Stateful: marked C2C cases only.',
                 color='#555555', fontsize=10)
        fig.savefig(destination, dpi=150, facecolor='white')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or args.json.with_suffix('.png')
    plot(json.loads(args.json.read_text()), destination)
    print(f'Chart: {destination}')


if __name__ == '__main__':
    main()
