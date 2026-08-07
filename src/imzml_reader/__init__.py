# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from ms_nexus_tools.api import data_convert

from .data_source import ImzMLDataSource


def main() -> None:
    partial_args = data_convert.ProcessArgs.parse_config("imzml_convert")
    process_args = data_convert.ProcessArgs.parse_interactive(
        "imzml_convert", args=partial_args.remaining_args, exclude=["config"]
    )
    process_args.data_source = ImzMLDataSource(process_args.in_path)
    data_convert.process(process_args, partial_args.config)
