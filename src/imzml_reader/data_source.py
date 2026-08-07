# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
import itertools

from contextlib import AbstractContextManager
from typing import Any, Callable, NamedTuple
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from pyimzml.ImzMLParser import ImzMLParser

from ms_nexus_tools.lib.data_source import (
    AbstractDataSource,
    DataShape,
    Axis,
    AxisType,
    MultiCOO,
    Chunk,
    Shape,
    UnknownAxisError,
)


class ImzMLDataSource(AbstractDataSource):
    def __init__(self, filename):
        self.filename = filename
        self.parser = ImzMLParser(self.filename)
        self.coords = np.array(self.parser.coordinates)

        self.min_coords = np.min(self.coords, axis=0)
        self.max_coords = np.max(self.coords, axis=0)
        shape = list(self.max_coords - self.min_coords + 1)
        if shape[-1] == 1:
            shape.pop(-1)
            self.has_z = False
        else:
            self.has_z = True

        if not self.has_z:
            self.coords = self.coords[:, 0:2]

        shape.append(np.max(self.parser.mzLengths))
        self.total_shape = tuple(shape)
        self.density = self.coords.shape[0] / np.prod(shape)

        ic(self.parser.imzmldict)
        # ic(self.parser.metadata.pretty())

        self.is_centroid = self.parser.spectrum_mode == "centroid"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.parser.__exit__(exc_type, exc_value, traceback)

    def instrament_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the instrament metadata.
        """
        return {}

    def experiment_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the experiment metadata.
        """
        return {}

    def shape(self) -> DataShape:
        """
        Return the shape of the data.
        """
        return DataShape(
            shape=self.total_shape, is_sparse=True, worst_case_density=self.density
        )

    def signal_type(self) -> npt.DTypeLike:
        """
        Returns the type for data.
        """
        return self.parser.intensityPrecision

    def output_chunks(self) -> dict[str, Shape]:
        """
        Returns the names and chunking priorities of the desired output array.
        For examlpe simple image data (x,y, spectra) with shape (32,32,184000)
        might produce:
        'images':   (1,1,2) -> (32,32,1)
        'spectra':  (2,2,1) -> (1,1,184000)
        """
        if self.has_z:
            return {
                "images": (1, 1, 1, 2),
                "spectra": (2, 2, 2, 1),
            }
        return {
            "images": (1, 1, 2),
            "spectra": (2, 2, 1),
        }

    def read_chunks(self) -> list[Shape] | None:
        """
        Retruns a list of chunking priorities taht can be used for reading the raw data.
        This is used in conjunction with chunk_read_count to select the most efficient reading scheme, and to track progress.
        If this returns None, then the priorities from the output_chunks are used.

        This is not an abstract method and defaults to returning None.
        """
        return [(2, 2, 1)] if not self.has_z else [(2, 2, 2, 1)]

    def chunk_read_count(self, memory_chunk: Shape) -> int:
        """
        Returns the number of read operations needed to fill the provided memory chunk.
        """
        return np.sum(memory_chunk[0:-1])

    def axis_definitions(self) -> list[Axis]:
        """
        Returns the axis that should be used when storing the data.
        """
        axes = []
        axes.append(Axis("x", 0, AxisType.EXACT, np.float32, "m"))
        axes.append(Axis("y", 1, AxisType.EXACT, np.float32, "m"))
        if self.has_z:
            axes.append(Axis("z", 2, AxisType.EXACT, np.float32, "m"))
            axes.append(
                Axis(
                    "mz",
                    3,
                    AxisType.BINNED if self.is_centroid else AxisType.EXACT,
                    self.parser.mzPrecision,
                    "mz",
                )
            )
        else:
            axes.append(
                Axis(
                    "mz",
                    2,
                    AxisType.BINNED if self.is_centroid else AxisType.EXACT,
                    self.parser.mzPrecision,
                    "mz",
                )
            )
        return axes

    def exact_axis_values(self, axis: Axis) -> np.ndarray:
        """
        Returns the values for the specified exact axis.
        """
        space_coords = ["x", "y", "z"]
        if axis.name in space_coords:
            mult = self.parser.imzmldict.get(f"pixel size {axis.name}", 1)
            inx = space_coords.index(axis.name)
            return np.arange(self.min_coords[inx], self.max_coords[inx] + 1) * mult
        if axis.name == "mz":
            # I am not sure what to do here! (plagues and curses, the reader does not provide this information!)
            # TODO: (dmd): Improve pyImzML to provide some data for this.
            # https://github.com/rosalindfranklininstitute/imzml-reader/issues/1
            return np.arange(0, self.total_shape[-1])
        raise UnknownAxisError(f"Axis {axis.name} is not a know axis.")

    def binned_axis_edges(self, axis: Axis) -> np.ndarray:
        """
        Returns the bin edges used to histogram the given binned axis.
        This is used for generting the output accumulations accros this axis, if required.
        """
        if axis.name == "mz":
            # I am not sure what to do here! (plagues and curses, the reader does not provide this information!)
            # TODO: (dmd): Improve pyImzML to provide some data for this.
            # https://github.com/rosalindfranklininstitute/imzml-reader/issues/1
            return np.arange(0, self.total_shape[-1])
        raise UnknownAxisError(f"Axis {axis.name} is not a know axis.")

    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        """
        Returns the names and lists of axis that should be
        accumulated (summed and max).
        For examlpe simple image data (x,y, spectra):
        might produce:
        'total_images':     ('mz') # Accumulate over the spectra
        'total_spectra':    ('x','y') # Accumulate over the images
        """
        accumulations = {}
        accumulations["total_images"] = "mz"
        if self.has_z:
            accumulations["total_spectra"] = ("x", "y", "z")
        else:
            accumulations["total_spectra"] = ("x", "y")
        return accumulations

    def fill_chunk(
        self,
        memory_chunk: Chunk,
        update: Callable[[int], None],
    ) -> np.ndarray | MultiCOO:
        """
        Read data from the source in the region specified by
        memory_chunk and return that data.

        Parameters:
        memory_chunk:   The bounds of the data to read.
        update:         A callback to update progress.
                        The total of the progress counter is
                        sum([chunk_read_count(mc) for mc in all_memory_chunks])
        Returns:
        The data from the source.
        -> return_data.shape == self.shape()

        """

        coord_mask = (self.coords[:, 0] >= memory_chunk[0].start) and (
            self.coords[:, 0] <= memory_chunk[0].stop
        )
        coord_mask &= (self.coords[:, 1] >= memory_chunk[1].start) and (
            self.coords[:, 1] <= memory_chunk[1].stop
        )
        if self.has_z:
            coord_mask &= (self.coords[:, 2] >= memory_chunk[2].start) and (
                self.coords[:, 2] <= memory_chunk[2].stop
            )
        coord_inx = np.nonzero(coord_mask)

        coords: list[list[int]] = []
        counts: list[int] = []
        int_data: list[np.ndarray] = []
        mz_data: list[np.ndarray] = []
        inx_data: list[np.ndarray] = []

        coord_parts = [memory_chunk.range(0), memory_chunk.range(1)]
        if self.has_z:
            coord_parts.append(memory_chunk.range(2))

        for inx in coord_inx:
            current_coords = self.coords[inx, :]
            mzs, ints = self.parser.getspectrum(current_coords)
            count = len(mzs)
            counts.append(count)
            coords.append([*current_coords, 0])
            int_data.append(ints)
            mz_data.append(mzs)

            # TODO (dmd): Can we find a way to get the actual axis values? See above.
            # https://github.com/rosalindfranklininstitute/imzml-reader/issues/1
            inx_data.append(np.arange(len(mzs)))

            update(1)

        mz_array = np.concatenate(mz_data)
        # TODO (dmd): Can we find a way to get the actual axis values? See above.
        # https://github.com/rosalindfranklininstitute/imzml-reader/issues/1
        labels = np.concatenate(inx_data)
        labels[labels == self.total_shape[-1]] = self.total_shape[-1] - 1
        final_coords = np.array(coords).T
        final_coords = np.repeat(final_coords, counts, axis=1)
        final_coords[3 if self.has_z else 2, :] = labels
        return MultiCOO(
            coords=final_coords,
            values={"signal": np.concatenate(int_data), "mz": mz_array},
        )
