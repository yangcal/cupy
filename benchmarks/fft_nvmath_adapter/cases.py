"""Pure case data for the warm FFT adapter comparison; no GPU imports.

For inverse real transforms, shape describes the full real output. The driver
generates a valid packed complex input. numpy_axes is materialized by the
driver after importing NumPy, so --list-cases remains CPU-only.
"""

from __future__ import annotations


def case(
    name,
    group,
    public,
    shape,
    *,
    dtype=None,
    kwargs=None,
    layout="C",
    minimal=False,
    numpy_axes=False,
    arange=False,
):
    return {
        "name": name,
        "group": group,
        "public": public,
        "shape": tuple(shape),
        "dtype": dtype
        or ("float32" if public.startswith("rfft") else "complex64"),
        "kwargs": dict(kwargs or {}),
        "layout": layout,
        "minimal": minimal,
        "numpy_axes": numpy_axes,
        "arange": arange,
    }


def full_kwargs(shape):
    return (
        {"axis": -1} if len(shape) == 1 else {"axes": tuple(range(len(shape)))}
    )


def basic():
    return [
        case(
            f"basic_{ndim}d",
            "basic",
            "fft" if ndim == 1 else "fftn",
            (2,) * ndim,
            kwargs=full_kwargs((2,) * ndim),
            minimal=True,
            arange=True,
        )
        for ndim in (1, 2, 3)
    ]


def sweep():
    result = []
    sizes = (
        ((2,), (8,), (64,), (1024,), (65536,), (1048576,)),
        ((2, 2), (8, 8), (32, 32), (128, 128), (1024, 1024)),
        ((2, 2, 2), (4, 4, 4), (16, 16, 16), (64, 64, 64), (128, 64, 128)),
    )
    for ndim, shapes in enumerate(sizes, 1):
        for shape in shapes:
            name = "x".join(map(str, shape))
            result.append(
                case(
                    f"c2c_{ndim}d_{name}",
                    "sizes",
                    "fft" if ndim == 1 else "fftn",
                    shape,
                    kwargs=full_kwargs(shape),
                    minimal=True,
                )
            )

    for ndim, shape in enumerate(((1024,), (32, 32), (16, 16, 16)), 1):
        for inverse, norms in (
            (False, ("backward", "forward", "ortho")),
            (True, (None, "forward", "ortho")),
        ):
            public = ("ifft" if inverse else "fft") + ("n" if ndim > 1 else "")
            for norm in norms:
                result.append(
                    case(
                        f"norm_{ndim}d_{public}_{norm}",
                        "norms",
                        public,
                        shape,
                        kwargs={**full_kwargs(shape), "norm": norm},
                    )
                )
        largest = sizes[ndim - 1][-1]
        result.append(
            case(
                f"inverse_largest_{ndim}d",
                "norms",
                "ifft" if ndim == 1 else "ifftn",
                largest,
                kwargs=full_kwargs(largest),
            )
        )

    patterns = (
        ("2d_single", "fftn", (3, 16), (16, 65536), (1,)),
        ("rank3_trailing", "fft2", (2, 8, 16), (4, 256, 512), None),
        ("2d_reordered", "fftn", (8, 16), (1024, 1024), (1, 0)),
        ("rank3_nonconsecutive", "fftn", (4, 3, 8), (128, 4, 1024), (0, 2)),
        ("rank3_list", "fftn", (4, 3, 8), (128, 4, 1024), [0, 2]),
        ("rank3_numpy_int64", "fftn", (4, 3, 8), (128, 4, 1024), (0, 2)),
        ("2d_mixed_sign", "fftn", (8, 16), (1024, 1024), (0, -1)),
        ("rank4_partial", "fftn", (2, 3, 8, 16), (2, 4, 256, 512), (2, 3)),
    )
    for name, public, small, large, axes in patterns:
        for size, shape in (("small", small), ("large", large)):
            result.append(
                case(
                    f"axes_{name}_{size}",
                    "axes",
                    public,
                    shape,
                    kwargs={} if axes is None else {"axes": axes},
                    numpy_axes="numpy_int64" in name,
                )
            )
    for size, shape in (("small", (2, 8, 16)), ("large", (4, 256, 512))):
        for norm in (None, "forward", "ortho"):
            result.append(
                case(
                    f"batched_ifft2_{size}_{norm}",
                    "axes",
                    "ifft2",
                    shape,
                    kwargs={"norm": norm},
                )
            )

    real_shapes = (
        ((32,), (1048576,)),
        ((8, 16), (1024, 1024)),
        ((4, 8, 16), (128, 64, 128)),
    )
    for suffix, shapes in zip(("", "2", "n"), real_shapes):
        for inverse in (False, True):
            public = ("irfft" if inverse else "rfft") + suffix
            for size, shape in zip(("small", "large"), shapes):
                result.append(
                    case(f"{public}_{size}", "real_double", public, shape)
                )
    for name, public, shape, dtype, norm in (
        ("rfft_odd33_forward", "rfft", (33,), "float32", "forward"),
        ("rfft_odd1m_ortho", "rfft", (1048575,), "float32", "ortho"),
        ("irfft2_small_forward", "irfft2", (8, 16), "complex64", "forward"),
        ("irfft2_large_ortho", "irfft2", (512, 1024), "complex64", "ortho"),
        ("double_fft_large", "fft", (1048576,), "complex128", None),
        ("double_ifft_medium", "ifft", (65536,), "complex128", None),
        ("double_rfftn_forward", "rfftn", (64, 64, 64), "float64", "forward"),
        ("double_irfftn_ortho", "irfftn", (64, 64, 64), "complex128", "ortho"),
    ):
        result.append(
            case(
                name,
                "real_double",
                public,
                shape,
                dtype=dtype,
                kwargs={"norm": norm},
            )
        )

    for public in ("fft", "ifft"):
        for layout, shapes in (
            ("F", ((8, 16), (16, 65536))),
            ("strided", ((64,), (1048576,))),
        ):
            for size, shape in zip(("small", "large"), shapes):
                result.append(
                    case(
                        f"layout_{layout}_{public}_{size}",
                        "layouts",
                        public,
                        shape,
                        layout=layout,
                    )
                )
    return result
