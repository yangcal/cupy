"""Render the warm comparison as compact tables, without GPU dependencies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


ROWS = ('stateful', 'minimal', 'original', 'axes', 'fused', 'c2r', 'ajay',
        'native')
HEADERS = ('Stateful', 'Minimal', 'Original', 'Axes', 'Fusion', 'C2R', 'Ajay',
           'Native')
GROUPS = (
    'C2R input preservation and normalization',
    'C2C normalization — contiguous operands',
    'R2C normalization',
    'Batches and axes variants',
    'C2C normalization — Fortran and strided operands',
)


def grouped_rows(cases):
    """Keep each operand's norms together, with scaled requests first."""
    families = defaultdict(lambda: defaultdict(list))
    for case in cases:
        spec = case['case']
        if spec['public'].startswith('irfft'):
            group = GROUPS[0]
        elif spec['layout'] != 'C':
            group = GROUPS[4]
        elif spec['public'].startswith('rfft'):
            group = GROUPS[2]
        elif spec.get('group') == 'axes':
            group = GROUPS[3]
        else:
            group = GROUPS[1]
        operand = (tuple(case['operand_shape']), spec['dtype'], spec['layout'])
        families[group][operand].append(case)

    result = []
    for group in GROUPS:
        operands = families.get(group, {})
        order = sorted(operands, key=lambda operand: (
            not any(row['normalization']['scaling_applies']
                    for row in operands[operand]),
            math.prod(operand[0]), operand))
        rows = [row for operand in order for row in sorted(
            operands[operand], key=lambda case: (
                not case['normalization']['scaling_applies'], case['case_id']))]
        if rows:
            result.append((group, rows))
    return result


def _times(case, metric):
    values = case['summary']['rows_' + metric + '_us']
    return [f'{values[row]:.3f}' if row in values else '—' for row in ROWS]


def _request(case):
    spec = case['case']
    # JSON represents tuple axes as lists. Preserve the measured call's axis
    # spelling, including list and NumPy-integer variants in the sweep.
    arguments = ', '.join(
        f"{name}={case['runtime_axes']}"
        if name in ('axis', 'axes') and 'runtime_axes' in case
        else f'{name}={value!r}'
        for name, value in spec['kwargs'].items())
    return f"{spec['public']}(a{', ' if arguments else ''}{arguments})"


def _table(cases, metric):
    lines = [
        '| ID | Request | Operand | ' + ' | '.join(HEADERS) + ' |',
        '| ---: | --- | --- | ' + ' | '.join(['---:'] * len(ROWS)) + ' |',
    ]
    for case in cases:
        spec = case['case']
        operand = f"{tuple(case['operand_shape'])} {spec['dtype']} {spec['layout']}"
        lines.append(
            f"| {case['case_id']} | `{_request(case)}` | `{operand}` | "
            + ' | '.join(_times(case, metric)) + ' |')
    return lines


def render(payload):
    """Produce CPU tables and a separate GPU-event table from measured rows."""
    if payload['status'] != 'complete':
        raise ValueError('Only a complete benchmark can be reported')
    env, config = payload['environment'], payload['configuration']
    node = env.get('node', env.get('host', {}).get('node'))
    machine = [env.get('gpu'), env.get('cpu_model'), node,
               'Python ' + env['python'].split()[0] if env.get('python') else None]
    lines = [
        '# Warm FFT comparison', '',
        ' · '.join(str(value) for value in machine if value), '',
        f"Median CPU times in µs: {config['blocks']} balanced blocks, "
        f"{config['repeats']} samples and {config['warmups']} warmups per row.", '',
        'Original uses the existing adapter. Axes adds common-axis handling and '
        'no-op scaling shortcuts, with the original C2R reset. Fusion adds '
        'NVRTC runtime-divisor scaling with semantic keys to Axes; C2R adds '
        'prepared input copying to Axes while retaining separate scaling. '
        'Ajay combines both changes. '
        'All nvmath rows share the execute/workspace candidate and dtype/stride '
        'helpers. Minimal and Stateful cover the marked forward C2C cases.', '',
    ]
    tiny = [case for case in payload['cases']
            if case['case'].get('minimal')
            and tuple(case['operand_shape']) in ((2,), (2, 2), (2, 2, 2))]
    if tiny:
        lines.extend([
            '## Summary', '',
            '| Operand | ' + ' | '.join(HEADERS) + ' |',
            '| --- | ' + ' | '.join(['---:'] * len(ROWS)) + ' |',
        ])
        for case in sorted(tiny, key=lambda case: len(case['operand_shape'])):
            lines.append(f"| `{tuple(case['operand_shape'])}` | "
                         + ' | '.join(_times(case, 'cpu')) + ' |')
        lines.append('')

    lines.extend(['## Full CPU results', ''])
    groups = grouped_rows(payload['cases'])
    for name, cases in groups:
        lines.extend(['### ' + name, '', *_table(cases, 'cpu'), ''])
    lines.extend([
        '<details>', '<summary>GPU-event times in µs</summary>', '',
        'These intervals include host submission gaps.', '',
    ])
    for name, cases in groups:
        lines.extend(['### ' + name, '', *_table(cases, 'gpu'), ''])
    lines.extend(['</details>', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or args.json.with_suffix('.md')
    destination.write_text(render(json.loads(args.json.read_text())))


if __name__ == '__main__':
    main()
