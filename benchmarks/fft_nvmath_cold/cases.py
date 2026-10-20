"""Curated cold-path coverage, without workload frequency weights."""

from __future__ import annotations


def catalog(warm_cases):
    cases = warm_cases.sweep()
    # Contrast smooth/power-of-two transforms with awkward and prime lengths.
    extra = (
        (127,),
        (1009,),
        (65537,),
        (31, 33),
        (127, 129),
        (7, 9, 11),
        (31, 33, 35),
    )
    for shape in extra:
        for real in (False, True):
            public = ("rfft" if real else "fft") + (
                "n" if len(shape) > 1 else ""
            )
            cases.append(
                warm_cases.case(
                    f"awkward_{public}_{'x'.join(map(str, shape))}",
                    "awkward",
                    public,
                    shape,
                )
            )
    # Even real output lengths with non-power-of-two factors for C2R.
    for shape in ((126,), (1008,), (65538,), (30, 34), (6, 10, 14)):
        public = "irfft" + ("n" if len(shape) > 1 else "")
        cases.append(
            warm_cases.case(
                f"awkward_{public}_{'x'.join(map(str, shape))}",
                "awkward",
                public,
                shape,
            )
        )
    return cases
