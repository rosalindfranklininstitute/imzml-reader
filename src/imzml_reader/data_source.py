# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, override

import numpy as np
import numpy.typing as npt
from pyimzml.ImzMLParser import ImzMLParser

from fastdigest import TDigest

from xartus.lib import (
    AbstractDataSource,
    DataShape,
    Axis,
    MultiCOO,
    Chunk,
    Shape,
    UnknownAxisError,
    Signal,
)
from xartus.lib.writers import AbstractWriter, SumAndMax


class ImzMLDataSource(AbstractDataSource):
    def __init__(self, filename, force_binned: bool = False):
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

        if force_binned:
            self.is_centroid = True
        else:
            self.is_centroid = self.parser.spectrum_mode == "centroid"

        digest = TDigest()

        for ii in range(len(self.parser.coordinates)):
            digest.batch_update(self.parser.getspectrum(ii)[0])

        mz_min = digest.min()
        mz_max = digest.max()
        self.mz_edges = np.array(
            digest.quantile_vec(
                np.linspace(0, 1, shape[-1] + (1 if self.is_centroid else 0))
            )
        )

        self.nnz = digest.mass()

    @override
    def __enter__(self):
        return self

    @override
    def __exit__(self, exc_type, exc_value, traceback):
        self.parser.__exit__(exc_type, exc_value, traceback)

    @override
    def instrument_metadata(self) -> dict[str, Any]:
        return {}

    @override
    def user_metadata(self) -> dict[str, Any]:
        return {}

    @override
    def experiment_metadata(self) -> dict[str, Any]:
        return {}

    @override
    def shape(self) -> DataShape:
        """
        Return the shape of the data.
        """
        return DataShape(shape=self.total_shape, worst_case_density=self.density)

    @override
    def signal_definition(self) -> Signal:
        """
        Returns the type for data.
        """
        return Signal(
            name="signal", dtype=self.parser.intensityPrecision, units="intensity"
        )

    @override
    def output_chunk_priorities(self) -> dict[str, Shape]:
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

    @override
    def read_chunks(self) -> list[Shape] | None:
        """
        Retruns a list of chunking priorities taht can be used for reading the raw data.
        This is used in conjunction with chunk_read_count to select the most efficient reading scheme, and to track progress.
        If this returns None, then the priorities from the output_chunks are used.

        This is not an abstract method and defaults to returning None.
        """
        return [(2, 2, 1)] if not self.has_z else [(2, 2, 2, 1)]

    @override
    def chunk_read_count(self, memory_chunk: Shape) -> int:
        """
        Returns the number of read operations needed to fill the provided memory chunk.
        """
        return np.sum(memory_chunk[0:-1])

    @override
    def axis_definitions(self) -> list[Axis]:
        """
        Returns the axis that should be used when storing the data.
        """
        axes = []
        axes.append(Axis("x", 0, np.int64, "m"))
        axes.append(Axis("y", 1, np.int64, "m"))
        if self.has_z:
            axes.append(Axis("z", 2, np.int64, "m"))
            axes.append(
                Axis(
                    "mz",
                    3,
                    np.int64,
                    "mz",
                )
            )
        else:
            axes.append(
                Axis(
                    "mz",
                    2,
                    np.float64,
                    "mz",
                )
            )
        return axes

    @override
    def axis_values(self, axis: Axis) -> np.ndarray:
        """
        Returns the values for the specified exact axis.
        """
        space_coords = ["x", "y", "z"]
        if axis.name in space_coords:
            mult = self.parser.imzmldict.get(f"pixel size {axis.name}", 1)
            inx = space_coords.index(axis.name)
            return np.arange(self.min_coords[inx], self.max_coords[inx] + 1) * mult
        if axis.name == "mz":
            return self.mz_edges
        raise UnknownAxisError(f"Axis {axis.name} is not a know axis.")

    @override
    def output_accumulations(self) -> list[AbstractWriter]:
        """
        Returns the names and lists of axis that should be
        accumulated (summed and max).
        For examlpe simple image data (x,y, spectra):
        might produce:
        'total_images':     ('mz') # Accumulate over the spectra
        'total_spectra':    ('x','y') # Accumulate over the images
        """
        accumulations = []
        accumulations.append(SumAndMax("total_images", axes=["mz"]))
        if self.has_z:
            accumulations.append(SumAndMax("total_spectra", axes=["x", "y", "z"]))
        else:
            accumulations.append(SumAndMax("total_spectra", axes=["x", "y"]))
        return accumulations

    @override
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
        coord_mask = self.coords[:, 0] >= memory_chunk[0].start
        coord_mask &= self.coords[:, 0] <= memory_chunk[0].stop
        coord_mask &= self.coords[:, 1] >= memory_chunk[1].start
        coord_mask &= self.coords[:, 1] <= memory_chunk[1].stop
        if self.has_z:
            coord_mask &= (self.coords[:, 2] >= memory_chunk[2].start) and (
                self.coords[:, 2] <= memory_chunk[2].stop
            )
        coord_inx = np.nonzero(coord_mask)[0]

        coords: list[list[int]] = []
        counts: list[int] = []
        int_data: list[np.ndarray] = []
        mz_data: list[np.ndarray] = []

        coord_parts = [memory_chunk.range(0), memory_chunk.range(1)]
        if self.has_z:
            coord_parts.append(memory_chunk.range(2))

        for inx in coord_inx:
            current_coords = self.coords[inx, :] - 1
            mzs, ints = self.parser.getspectrum(inx)
            count = len(mzs)
            counts.append(count)
            coords.append([*current_coords, 0])
            int_data.append(ints)
            mz_data.append(mzs)

            update(1)

        mz_array = np.concatenate(mz_data)
        labels = np.searchsorted(self.mz_edges[1:], mz_array)
        labels[labels == self.total_shape[-1]] = self.total_shape[-1] - 1
        final_coords = np.array(coords).T
        final_coords = np.repeat(final_coords, counts, axis=1)
        final_coords[3 if self.has_z else 2, :] = labels

        # return MultiCOO(
        #     coords=final_coords,
        #     values={"signal": np.concatenate(int_data), "mz": mz_array},
        # )

        values = {"signal": np.concatenate(int_data)}

        return MultiCOO(
            coords=final_coords,
            values=values,
        )
