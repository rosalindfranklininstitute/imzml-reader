# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from xartus.api import data_convert
from argsui import arg_field, FilePathType
from dataclasses import dataclass

from .data_source import ImzMLDataSource

logging.basicConfig(format="%(module)s: %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger(data_convert.__name__).setLevel(logging.INFO)


@dataclass
class ProcessArgs(data_convert.ProcessArgs):
    in_path: Path = arg_field(
        "--input",
        doc="The name of the folder to import data from.",
        type=FilePathType(must_exist=True),
        default=None,
    )


def main() -> None:
    partial_args = ProcessArgs.parse_config("imzml_convert")
    process_args = ProcessArgs.parse_interactive(
        "imzml_convert", args=partial_args.remaining_args, exclude=["config"]
    )
    process_args.data_source = ImzMLDataSource(process_args.in_path)
    data_convert.process(process_args, partial_args.config)
