#!/home/nilsm/miniconda3/envs/analysis/bin/python3
"""Utility to convert SDF-files to HDF5-format.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os import process_cpu_count
from pathlib import Path
import tarfile
from threading import RLock
from typing import List, Any

import h5py
from tqdm import tqdm
import sdf_helper as sdf

IGNORED_ATTRIBUTES = [
    "blocklist", "datatype", "dims", "data_length", "grid_mid", "grid"
]

def addAttribute(
    h5_obj: h5py.File | h5py.Group | h5py.Dataset,
    attr_name: str,
    attr_value: Any
) -> None:
    """Adds HDF5-attribute to the output file and converts data types as necessary.

    Args:
        h5_obj (h5py.File | h5py.Group | h5py.Dataset): The HDF5-object that should get the
            attribute.
        attr_name (str): name of the attribute
        attr_value (Any): value of the attribute
    """
    if isinstance(attr_value, str):
        h5_obj.attrs.create(
            name=attr_name,
            data=attr_value,
            dtype=h5py.string_dtype()
        )
    elif isinstance(attr_value, float) or isinstance(attr_value, int):
        h5_obj.attrs.create(
            name=attr_name,
            data=attr_value
        )
    elif isinstance(attr_value, tuple):
        assert len(attr_value) > 0, "Expected non-zero length tuple"
        if isinstance(attr_value[0], str):
            h5_obj.attrs.create(
                name=attr_name,
                data=attr_value,
                shape=len(attr_value),
                dtype=h5py.string_dtype()
            )
        else:
            try:
                h5_obj.attrs.create(
                    name=attr_name,
                    data=attr_value
                )
            except:
                print(f"Could not create tuple-attribute '{attr_name}' for block '{h5_obj.name}'")
    else:
        print(f"Unexpected type '{type(attr_value)}' for attribute '{attr_name}'")

def saveNativeBlock(
    h5_file: h5py.File,
    block: sdf.sdf.BlockPlainVariable|sdf.sdf.BlockPlainMesh|sdf.sdf.BlockConstant,
    compression_level: int,
) -> None:
    """Save a block from the SDF-format to the HDF5-file.

    Args:
        h5_file (h5py.File): HDF5-file handle
        block (BlockPlainVariable|BlockPlainMesh|BlockConstant): SDF-block that should be saved
        compression_level (int): gzip compression level (0-9)
    """
    try:
        block_data = block.data
    except:
        print(f"WARNING: Could not read data-block from {block.name}")
        return
    assert hasattr(block, "name"), "Expected name attribute"

    object_name = Path(block.name).with_stem(Path(block.id).name).as_posix()
    if hasattr(block, "labels") and len(block_data) > 1:  # assumes that that all grids are converted to tuples
        assert isinstance(block, sdf.sdf.BlockPlainMesh), "Expected mesh"
        h5_obj = h5_file.create_group(name=object_name)
        for idx, axis_label in enumerate(block.labels):
            dataset = h5_obj.create_dataset(
                name=axis_label, data=block_data[idx],
                compression="gzip", compression_opts=compression_level
            )
            for attr_name in ["extents", "mult", "units"]:
                if hasattr(block, attr_name):
                    attr_data = getattr(block, attr_name)[idx]
                    addAttribute(dataset, attr_name, attr_data)
            ignored_names = IGNORED_ATTRIBUTES + ["data", "extents", "mult", "units"]
            for attr_name in dir(block):
                if attr_name.startswith("__") or attr_name in ignored_names:
                    continue
                if not hasattr(block, attr_name):
                    continue

                attr_value = getattr(block, attr_name)
                addAttribute(dataset, attr_name, attr_value)

    else:
        ignored_names = IGNORED_ATTRIBUTES + ["data"]
        try:
            dataset = h5_file.create_dataset(name=object_name, data=block_data)
        except:
            print(f"WARNING: Could not create {object_name} dataset. Skipped!")
            return
        for attr_name in dir(block):
            if attr_name.startswith("__") or attr_name in ignored_names:
                continue
            if not hasattr(block, attr_name):
                continue

            attr_value = getattr(block, attr_name)
            addAttribute(dataset, attr_name, attr_value)

def saveInfoBlock(h5_file: h5py.File, block_name: str, block: dict) -> None:
    """Save an info block from the SDF-file to HDF5.

    Args:
        h5_file (h5py.File): HDF5-file to save
        block_name (str): name of the info-block
        block (dict): info-block from SDF
    """
    group = h5_file.create_group(block_name)
    group.attrs.update(block)

def saveSDFFileToHDF5(sdf_file_path: Path, h5_file_path: Path, compression_level: int) -> None:
    """Save an SDF-file to HDF5-format.

    Args:
        sdf_file_path (Path): filepath to the SDF-file
        h5_file_path (Path): filepath to the HDF5-file
        compression_level (int): gzip compression level (0-9)
    """
    assert sdf_file_path.exists(), "File should exist"
    assert 0 <= compression_level <= 9, "Compression level must be integer 0-9"
    sdf_data = sdf.getdata(sdf_file_path.as_posix(), verbose=False)
    sdf_blocks = [x for x in dir(sdf_data) if not x.startswith("__")]

    with h5py.File(h5_file_path, mode="w") as h5_file:
        # save each 'block' in the file. These are the quantities/variables/grids
        for block_name in sdf_blocks:
            # NOTE: mid-points are never saved
            if block_name.endswith("_mid"):
                continue
            block = getattr(sdf_data, block_name)
            if isinstance(block, dict):
                saveInfoBlock(h5_file, block_name, block)
            else:
                saveNativeBlock(h5_file, block, compression_level)

def archiveSDFFiles(
    sdf_files: List[Path],
    tar_archive: Path
) -> None:
    """Creates a TAR-archive of all SDF-files.

    Args:
        sdf_files (List[Path]): list of the SDF-files to add to the archive
        tar_archive (Path): name and filepath of the tar-archive
    """
    with tarfile.open(tar_archive, mode="w:xz") as tar:
        for file_path in tqdm(sdf_files, desc="Files", leave=False):
            tar.add(name=file_path, arcname=file_path.name, recursive=False)

def convertDirectory(
    data_directory: Path,
    pool: ThreadPoolExecutor,
    hdf5_subdirectory: str="",
    overwrite: bool=False,
    compression_level: int=4,
) -> None:
    """Processes a directory of SDF-files and covert them to HDF5-format.
    The old files are saved to a tar-archive and then delete.

    Args:
        data_directory (Path): directory that contains the SDF-files
        pool (ThreadPoolExecutor): Process pool to use.
        hdf5_subdirectory (str, optional): If the HDF5-files should be created in a sub-directory.
            Defaults to "".
        archive_name (str, optional): Name of the tar-archive. Defaults to
            "original_sdf-files.tar.xz".
        overwrite (bool, optional): Whether to overwrite any existing files. Defaults to False.
            This skips thexisting files.
        compression_level (int, optional): gzip compression level (0-9). Defautls to 4.
    """
    output_folder = data_directory / hdf5_subdirectory
    output_folder.mkdir(exist_ok=True, parents=False)
    sdf_files = sorted(data_directory.glob("*.sdf", case_sensitive=False))
    h5_files = [(output_folder / file.name).with_suffix(".h5") for file in sdf_files]
    filtered_files = list(filter(
        lambda files: not files[1].exists() or overwrite,
        zip(sdf_files, h5_files)
    ))
    if len(filtered_files) == 0:
        return
    futures = pool.map(
        partial(saveSDFFileToHDF5, compression_level=compression_level),
        *zip(*filtered_files)
    )
    list(tqdm(futures, total=len(filtered_files), desc="Files", leave=False))

def archiveDirectory(
    data_directory: Path,
    overwrite: bool=False,
    archive_name: str="original_sdf-files.tar.xz"
):
    """Archives SDF-files in directory.

    Args:
        data_directory (Path): Directory that contains the SDF-files
        overwrite (bool, optional): Whether to overwrite an existing archive. Defaults to False.
        archive_name (str, optional): Name of the archive file. Defaults to "original_sdf-files.tar.xz".
    """
    sdf_files = sorted(data_directory.glob("*.sdf", case_sensitive=False))
    archive_path = data_directory / archive_name
    if not archive_path.exists() or overwrite:
        archiveSDFFiles(sdf_files, archive_path)
        for sdf_file_path in sdf_files:
            sdf_file_path.unlink()

def _validatePathArgument(arg: str) -> Path:
    """Validates the argument of the given path.

    Args:
        arg (str): filepath

    Returns:
        Path: valid file path
    """
    path = Path(arg)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory '{path}' does not exist")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory")
    return path

def _validateNumProcs(arg: str) -> int:
    if not arg.isdigit():
        raise argparse.ArgumentTypeError("Supply a positive number of processes")
    n_procs = int(arg)
    if n_procs <= 0:
        raise argparse.ArgumentTypeError("Need at least one process")
    return n_procs

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="h5vstack",
        description="Converts all SDF-files in a folder to HDF5-files. "
            "SDF-files are then archived and deleted. Grid mid-points are not saved."
    )
    arg_parser.add_argument(
        "-r", "--recursive",
        help="Recursively convert SDF-files in all sub-directories. Defaults to off.",
        action="store_true")
    arg_parser.add_argument(
        "--compression",
        help="Compression level for the HDF5-datasets. Defaults to 4.",
        type=int, choices=range(10), default=4, metavar="0-9"
    )
    arg_parser.add_argument(
        "-p", "--procs",
        help="Number of processors to use. Defauts to the number of CPU-threads.",
        type=_validateNumProcs, default=process_cpu_count(), metavar="N_PROCS>=1"
    )
    arg_parser.add_argument(
        "--overwrite",
        help="If set will overwrite existing files/archives otherwise these wil be skipped.",
        action="store_true"
    )
    arg_parser.add_argument(
        "--no-archive",
        help="Skips archiving SDF-files.",
        action="store_true"
    )
    arg_parser.add_argument(
        "FOLDER",
        help="Location of the simulation-data",
        type=_validatePathArgument, nargs='+'
    )
    args = arg_parser.parse_args()

    if args.recursive:
        directories = sorted(sub_folder for folder in args.FOLDER for sub_folder in folder.glob("**"))
    else:
        directories = args.FOLDER
    directories = list(filter(lambda x: len(list(x.glob("*.sdf"))) > 0, directories))
    if len(directories) == 0:
        print("WARNING: No SDF-files found")
    else:
        tqdm.set_lock(RLock())
        with ThreadPoolExecutor(
            max_workers=args.procs,
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),)
        ) as pool:
            print("Creating HDF5-documents...")
            for folder in tqdm(directories, desc="Directories"):
                convertDirectory(
                    folder, pool, overwrite=args.overwrite,
                    compression_level=args.compression
                )
            if not args.no_archive:
                print("Archiving SDF-files...")
                futures = pool.map(partial(archiveDirectory, overwrite=args.overwrite), directories)
                list(tqdm(futures, desc="Directory", total=len(directories)))
        print("Done.")
