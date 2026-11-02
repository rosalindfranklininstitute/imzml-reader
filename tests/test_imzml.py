# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np

from xartus.api import data_convert
from xartus.lib.data_source import Axis
from imzml_reader import imzml_write_from_nexus as imzml

from pyimzml.ImzMLParser import ImzMLParser

from nexus_pixel_man_test_data import ManData, Man2DDataSource, data_files

import pytest


@pytest.fixture(scope="module")
def man_data() -> ManData:
    return ManData()


@pytest.fixture(scope="module")
def man_file() -> Path:
    return data_files()["man1"]


@pytest.fixture
def nx_file():
    filename = Path(__file__).parent / "test.nxs"
    if filename.exists():
        filename.unlink()
    yield filename
    filename.unlink()


@pytest.fixture
def imzml_files():
    filename = Path(__file__).parent / "test.imzML"
    dbname = Path(__file__).parent / "test.ibd"
    if filename.exists():
        filename.unlink()
    if dbname.exists():
        dbname.unlink()
    yield filename, dbname
    if filename.exists():
        filename.unlink()
    if dbname.exists():
        dbname.unlink()


def test_full_convert(nx_file, imzml_files, man_data, man_file):

    man_data_source = Man2DDataSource(man_data, multipliers={"mz": 0.1})

    convert_args = data_convert.ProcessArgs(
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(convert_args, {})
    assert nx_file.exists()

    imzml_args = imzml.ProcessArgs(
        in_path=nx_file,
        out_path=imzml_files[0],
        group_name="data",
        entry_name="images",
        signal="signal",
        mass="mz",
        x_axis=0,
        y_axis=1,
        z_axis=-1,
        mz_axis=2,
        one_indexed=False,
    )
    imzml.process(imzml_args, {})
    assert imzml_files[0].exists()
    assert imzml_files[1].exists()

    with ImzMLParser(filename=imzml_files[0]) as imzml_data:
        for ii, coords in enumerate(imzml_data.coordinates):
            mz_values, int_values = imzml_data.getspectrum(ii)
            np.testing.assert_allclose(man_data.dense[*coords[0:2], :], int_values[:])
            np.testing.assert_allclose(
                mz_values, (np.arange(man_data.shape[2]) * 0.1), atol=1e-5
            )


def test_full_convert_one_indexed(nx_file, imzml_files, man_data, man_file):

    man_data_source = Man2DDataSource(man_data)

    convert_args = data_convert.ProcessArgs(
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(convert_args, {})
    assert nx_file.exists()

    imzml_args = imzml.ProcessArgs(
        in_path=nx_file,
        out_path=imzml_files[0],
        group_name="data",
        entry_name="images",
        signal="signal",
        mass="mz",
        x_axis=0,
        y_axis=1,
        z_axis=-1,
        mz_axis=2,
        one_indexed=True,
    )
    imzml.process(imzml_args, {})
    assert imzml_files[0].exists()
    assert imzml_files[1].exists()

    with ImzMLParser(filename=imzml_files[0]) as imzml_data:
        for ii, coords in enumerate(imzml_data.coordinates):
            mz_values, int_values = imzml_data.getspectrum(ii)
            coords = tuple(cc - 1 for cc in coords)
            np.testing.assert_allclose(man_data.dense[*coords[0:2], :], int_values[:])


def test_swap_x_y(nx_file, imzml_files, man_data, man_file):

    man_data_source = Man2DDataSource(man_data)

    convert_args = data_convert.ProcessArgs(
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(convert_args, {})
    assert nx_file.exists()

    imzml_args = imzml.ProcessArgs(
        in_path=nx_file,
        out_path=imzml_files[0],
        group_name="data",
        entry_name="images",
        signal="signal",
        mass="mz",
        x_axis=1,
        y_axis=0,
        z_axis=-1,
        mz_axis=2,
        one_indexed=False,
    )
    imzml.process(imzml_args, {})
    assert imzml_files[0].exists()
    assert imzml_files[1].exists()

    with ImzMLParser(filename=imzml_files[0]) as imzml_data:
        for ii, coords in enumerate(imzml_data.coordinates):
            mz_values, int_values = imzml_data.getspectrum(ii)
            np.testing.assert_allclose(
                man_data.dense[coords[1], coords[0], :], int_values[:]
            )


def test_total_image(nx_file, imzml_files, man_data, man_file):

    man_data_source = Man2DDataSource(man_data)

    convert_args = data_convert.ProcessArgs(
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(convert_args, {})
    assert nx_file.exists()

    imzml_args = imzml.ProcessArgs(
        in_path=nx_file,
        out_path=imzml_files[0],
        group_name="accumulations",
        entry_name="total_image",
        signal="signal",
        mass="y",
        x_axis=1,
        y_axis=-1,
        z_axis=0,
        mz_axis=2,
        one_indexed=False,
    )
    imzml.process(imzml_args, {})
    assert imzml_files[0].exists()
    assert imzml_files[1].exists()

    sum_image = np.sum(man_data.dense, axis=2)
    max_image = np.max(man_data.dense, axis=2)

    with ImzMLParser(filename=imzml_files[0]) as imzml_data:
        for ii, coords in enumerate(imzml_data.coordinates):
            mz_values, int_values = imzml_data.getspectrum(ii)
            if coords[2] == 0:
                np.testing.assert_allclose(sum_image[coords[0], :], int_values[:])
            elif coords[2] == 1:
                np.testing.assert_allclose(max_image[coords[0], :], int_values[:])
