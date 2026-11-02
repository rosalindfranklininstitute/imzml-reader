# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

import itertools
from typing import Any
from pathlib import Path
from dataclasses import dataclass

from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.compression import NoCompression

from tqdm import tqdm
import numpy as np

from argsui import arg_field, FilePathType, ConfigFileArgs, InteractiveArgs, ArgType

from xartus.lib.nxs import NexusFile


@dataclass
class ProcessArgs(ConfigFileArgs, InteractiveArgs):
    in_path: Path = arg_field(
        "-i",
        "--input",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The nxs file to process.",
        default=None,
        type=FilePathType(must_exist=True),
    )

    out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The output filename.",
        default=None,
        type=FilePathType(must_exist=False),
    )

    group_name: str = arg_field(
        require=True,
        help="The name of the subentry to extract data from.",
        default=None,
    )

    entry_name: str = arg_field(
        require=True,
        help="The name of the NXdata to extract data from.",
        default=None,
    )

    signal: str = arg_field(
        default="signal", help="The name of the dataset storing the intensities."
    )
    mass: str = arg_field(
        default="mz", help="The name of the dataset storing the mz values."
    )

    x_axis: int = arg_field(
        default=0,
        help="The dimension that represents the x-axis. Should be 0 or greater.",
    )

    y_axis: int = arg_field(
        default=1,
        help="The dimension that represents the y-axis. If set to a negaive value the data will be assumed to not have a y dimension and the y values will all be set to 0 (or 1).",
    )

    z_axis: int = arg_field(
        default=-1,
        help="The dimension that represents the z-axis. If set to a negaive value the data will be assumed to not have a z dimension and the z values will all be set to 0 (or 1).",
    )
    mz_axis: int = arg_field(
        default=-1,
        help="The dimension that represents the mz-axis. Use -1 for the last dimension.",
    )

    one_indexed: bool = arg_field(
        action="store_true",
        doc="If present the data will be written with indexing starting at 1.",
    )


def process(args: ProcessArgs, config: dict[str, Any] = {}) -> None:

    nx_file = NexusFile(args.in_path, mode="r")

    with nx_file.as_context():
        shape = nx_file.entry[args.group_name][args.entry_name][args.signal].shape
        mz_shape = nx_file.entry[args.group_name][args.entry_name][args.mass].shape
        if mz_shape != shape:
            mz = nx_file.entry[args.group_name][args.entry_name][args.mass][:]
        else:
            mz = None

        ndim = len(shape)
        if args.x_axis < 0 or args.x_axis >= ndim:
            raise ValueError(
                f"The x_axis dimension must be specified. It must be between {0} and {ndim}"
            )
        if args.mz_axis < -1 or args.mz_axis >= ndim:
            raise ValueError(
                f"The mz_axis dimension must be specified. It must be between {-1} and {ndim}"
            )

        if (
            args.x_axis == args.mz_axis
            or (
                args.y_axis >= 0
                and (args.y_axis == args.x_axis or args.y_axis == args.mz_axis)
            )
            or (
                args.z_axis >= 0
                and (args.z_axis == args.x_axis or args.z_axis == args.mz_axis)
            )
            or (args.y_axis >= 0 and args.z_axis >= 0 and args.y_axis == args.z_axis)
        ):
            raise ValueError(
                "The dimensions specified for the four axis (x,y,z, mz) should be unique."
            )

        required_ndim = 4
        if args.y_axis < 0:
            required_ndim -= 1
        elif args.y_axis >= ndim:
            raise ValueError(f"The y_axis dimension must be between {0} and {ndim}")

        if args.z_axis < 0:
            required_ndim -= 1
        elif args.z_axis >= ndim:
            raise ValueError(f"The z_axis dimension must be between {0} and {ndim}")

        if required_ndim != ndim:
            raise ValueError(
                f"Expected the data to have {required_ndim} dimensions, but found {ndim}."
            )

        with ImzMLWriter(
            args.out_path,
            mz_dtype=np.float32,
            intensity_dtype=np.float32,
            mode="continuous",
            mz_compression=NoCompression(),
            intensity_compression=NoCompression(),
            spec_type="profile",
        ) as writer:
            xyz = [
                (x, y, z)
                for x, y, z in itertools.product(
                    range(shape[args.x_axis]),
                    range(shape[args.y_axis]) if args.y_axis >= 0 else [0],
                    range(shape[args.z_axis]) if args.z_axis >= 0 else [0],
                )
            ]
            for coords in tqdm(xyz, desc="Writing imzML"):
                fill = 1 if args.one_indexed else 0
                result_inx: list[int | slice] = [fill] * required_ndim
                result_inx[args.x_axis] = coords[0]
                result_inx[args.mz_axis] = slice(None)
                if args.y_axis >= 0:
                    result_inx[args.y_axis] = coords[1]
                if args.z_axis >= 0:
                    result_inx[args.z_axis] = coords[2]

                intensity = nx_file.entry[args.group_name][args.entry_name][
                    args.signal
                ][*result_inx].astype(np.float32)
                if mz is None:
                    masses = nx_file.entry[args.group_name][args.entry_name][args.mass][
                        *result_inx
                    ].astype(np.float32)
                else:
                    masses = mz

                if args.one_indexed:
                    coords = tuple(cc + 1 for cc in coords)
                writer.addSpectrum(
                    masses,
                    intensity,
                    coords,
                )
