#!/home/nilsm/miniconda3/envs/analysis/bin/python3
"""Utility to combine many HDF5-files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from os import process_cpu_count
from pathlib import Path
from threading import RLock
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

INFO_GROUPS = ["/Run_info", "/Header"]
CONSTANT_GROUPS = ["/Grid"]

""" ---------------------------------------------------------------------------
Initialization
--------------------------------------------------------------------------- """

def createAttributeStore(src_file: h5py.File, num_files: int) -> dict[str, Any]:
    assert num_files > 0, "Number of files must positive."
    # Create a dict with keys Run_info/<attr_key>.
    # Values should be numpy array or list of appropiate size [None] * time.size
    attr_store = {}
    for group_name in INFO_GROUPS:
        assert group_name in src_file, f"Info group '{group_name}' does not exist."
        h5_group = src_file[group_name]
        assert len(h5_group.keys()) == 0, f"Info group '{group_name}' cannot contain objects."
        for attr_name, attr_value in h5_group.attrs.items():
            # build unique key
            key = f"{group_name}/{attr_name}"
            assert key not in attr_store, f"'{key}' already in attribute store."
            # NOTE: Assume that value is string of numpy-like type.
            if isinstance(attr_value, str):
                attr_store[key] = [None] * num_files
            else:
                assert hasattr(attr_value, "dtype") and hasattr(attr_value, "shape"), "Expected numpy-like type"
                attr_store[key] = np.empty(shape=(num_files, *attr_value.shape), dtype=attr_value.dtype)
    return attr_store

def createTargetDatasets(src_file: h5py.File, target_file: h5py.File, num_files: int) -> None:
    assert target_file.mode == "r+", "Write permission on target-file required."
    assert num_files > 0, "Number of files must positive."
    # Recursively go through and create groups/datasets not in 'Header' or 'Run_info' or 'Grid'.
    # Assert that the groups do not have attributes. Copy the attributes from the other datasets
    def _createDatasets(name: str, h5_obj: h5py.Group|h5py.Dataset):
        if any(name.startswith(group_name.removeprefix("/")) for group_name in INFO_GROUPS + CONSTANT_GROUPS):
            return
        if isinstance(h5_obj, h5py.Group):
            assert len(h5_obj.attrs.keys()) == 0, f"Group '{name}' is not allowed to have attributes."
            return
        src_ds = h5_obj
        target_ds = target_file.create_dataset(
            name=name,
            shape=(num_files,*src_ds.shape),
            dtype=src_ds.dtype,
        )
        target_ds.attrs.update(src_ds.attrs)

    src_file.visititems(_createDatasets)

def copyGridDatasets(src_file: h5py.File, target_file: h5py.File):
    assert target_file.mode == "r+", "Write permission on target-file required."
    # Copy datasets in 'Grid' to new file as they are
    for group_name in CONSTANT_GROUPS:
        src_file.copy(
            source=group_name,
            dest=target_file,
            shallow=False
        )

""" ---------------------------------------------------------------------------
Data transfer
--------------------------------------------------------------------------- """

def copyDatasets(file_idx: int, src_file: h5py.File, target_file: h5py.File):
    assert target_file.mode == "r+", "Write permission on target-file required."
    # For each file copy all datasets not in 'Header' or 'Run_info' or 'Grid'
    # assert that the attributes are the same and groups do not have attributes
    # assert that the dimensions are the same
    def _copyData(name: str, h5_obj: h5py.Group|h5py.Dataset):
        # skip special paths
        if any(name.startswith(group_name.removeprefix("/")) for group_name in INFO_GROUPS + CONSTANT_GROUPS):
            return
        # check conformity
        assert name in target_file, f"Object '{name}' does not exist on target-file."
        if isinstance(h5_obj, h5py.Group):
            assert len(h5_obj.attrs.keys()) == 0, f"Group '{name}' is not allowed to have attributes."
            return
        src_ds = h5_obj
        target_ds = target_file[name]
        assert src_ds.shape == target_ds.shape[1:], f"Dimensions ({src_ds.shape} vs. {target_ds.shape[1:]} of '{name}' in '{src_ds.file.filename}' do not match."
        assert src_ds.attrs == target_ds.attrs, f"Attributes of '{name}' in '{src_ds.file.filename}' do not match."
        # write data into target-file
        target_ds[file_idx] = src_ds

    src_file.visititems(_copyData)

def copyAttributes(file_idx: int, src_file: h5py.File, attr_store: dict[str, Any]) -> None:
    # For each attribute of 'Header and 'Run_info' write to the dict.
    # Assert that the key in the dict exists and the type is the same
    # Assert that these groups do not have dataset in them
    for group_name in INFO_GROUPS:
        # check conformity
        src_fname = src_file.filename
        assert group_name in src_file, f"Info group '{group_name}' in '{src_fname}' does not exist."
        h5_group = src_file[group_name]
        assert len(h5_group.keys()) == 0, f"Info group '{group_name}' in '{src_fname}' cannot contain objects."
        # add attributes to the store
        for attr_name, attr_value in h5_group.attrs.items():
            # build unique key
            key = f"{group_name}/{attr_name}"
            assert key in attr_store, f"Unknown attribute '{key}' in '{src_fname}'."
            # NOTE: Assume that value is string of numpy-like type.
            if isinstance(attr_value, str):
                assert isinstance(attr_store[key], list), f"Attribute '{key}' in '{src_fname}' has type string. Expected array ({attr_store[key].dtype})."
            else:
                assert hasattr(attr_value, "dtype") and hasattr(attr_value, "shape"), f"Attribute '{key}' in '{src_fname}' has type '{type(attr_value)}'. Expected numpy-like type."
                assert isinstance(attr_store[key], np.ndarray), f"Attribute '{key}' in '{src_fname}' has type {type(attr_value)}. Expected array ({attr_store[key].dtype})."
                assert attr_value.dtype == attr_store[key].dtype, f"Attribute '{key}' in '{src_fname}' has type {attr_value.dtype}. Expected array ({attr_store[key].dtype})."
                assert attr_value.shape == attr_store[key].shape[1:], f"Dimensions ({attr_value.shape} vs. {attr_store[key].shape}) of attribute '{key}' in '{src_fname}' do not match."
            # copy data into store
            attr_store[key][file_idx] = attr_value

""" ---------------------------------------------------------------------------
Store attributes
--------------------------------------------------------------------------- """

def writeAttributeStore(attr_store: dict[str, Any], target_file: h5py.File) -> None:
    # check whether a given item changes over time
    # changed -> Save as dataset with path given by key
    # unchanged -> Save first item as attribute as <group>.../<attribute-name>
    # For string data use:
    # ds = f.create_dataset('vlen_strings2', shape=4, dtype=h5py.string_dtype())
    # ds[:] = string_data
    for name, dataset in attr_store.items():
        group_name = Path(name).parent.as_posix()
        attr_name = Path(name).name
        h5_group = target_file.require_group(group_name)

        if all(np.all(dataset[0] == entry) for entry in dataset):
            assert attr_name not in h5_group.attrs, f"Attribute '{attr_name}' already exists in target file."
            h5_group.attrs[attr_name] = dataset[0]
        else:
            assert name not in target_file, f"Dataset '{name}' already exists in target file."
            # NOTE: Do we need to explicitly set dtype=h5py.string_dtype()?
            h5_group[attr_name] = dataset

""" ---------------------------------------------------------------------------
Main functions
--------------------------------------------------------------------------- """

def combineHDF5FilesInDirectory(src_directory: Path, target_file: Path) -> None:
    assert src_directory.is_dir(), f"Source '{src_directory}' not a directory"
    assert src_directory.exists(), f"Source '{src_directory}' does not exist."
    assert not target_file.exists(), f"Target file will not be overwritten."

    # initial setup
    src_files = sorted(src_directory.glob("*.h5"))
    assert len(src_files) > 0, "Found no src-files."
    num_files = len(src_files)
    with h5py.File(src_files[0]) as h5_src:
        with h5py.File(target_file, mode="x") as h5_target:
            attribute_store = createAttributeStore(h5_src, num_files)
            copyGridDatasets(h5_src, h5_target)
            createTargetDatasets(h5_src, h5_target, num_files)
    # transfer files
    with h5py.File(target_file, mode="r+") as h5_target:
        for file_idx, file in enumerate(tqdm(src_files, "Reading files", leave=False)):
            with h5py.File(file) as h5_src:
                copyAttributes(file_idx, h5_src, attribute_store)
                copyDatasets(file_idx, h5_src, h5_target)

    # write attribute-store to target-file
    with h5py.File(target_file, mode="r+") as h5_target:
        writeAttributeStore(attribute_store, h5_target)

def _validateSrcPathArgument(arg: str) -> Path:
    """Validates the argument of the given path.

    Args:
        arg (str): filepath

    Returns:
        Path: valid file path
    """
    src_path = Path(arg)
    if not src_path.exists():
        raise argparse.ArgumentTypeError(f"Directory '{src_path}' does not exist")
    if not src_path.is_dir():
        raise argparse.ArgumentTypeError(f"'{src_path}' is not a directory")
    return src_path

def validateNumProcs(arg: str) -> int:
    if not arg.isdigit():
        raise argparse.ArgumentTypeError("Supply a positive number of processes")
    n_procs = int(arg)
    if n_procs <= 0:
        raise argparse.ArgumentTypeError("Need at least one process")
    return n_procs

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-r", "--recursive",
        help="Recursively convert SDF-files in all sub-directories. Defaults to off.",
        action="store_true"
    )
    arg_parser.add_argument(
        "-p", "--procs",
        help="Number of processors to use. Defauts to the number of CPU-threads.",
        type=validateNumProcs, default=process_cpu_count(), metavar=">=1"
    )
    # TODO: Compression would be nice but it cannot be applied while copying otherwise things take forever.
    # arg_parser.add_argument(
    #     "--compression",
    #     help="Compression level for the HDF5-datasets. Defaults to 4.",
    #     type=int, choices=range(10), default=4, metavar="0-9"
    # )
    # arg_parser.add_argument(
    #     "--overwrite",
    #     help="If set will overwrite existing files/archives otherwise these wil be skipped.",
    #     action="store_true"
    # )
    arg_parser.add_argument(
        "source_directory",
        help="Location of the simulation-data (HDF5)",
        type=_validateSrcPathArgument
    )
    args = arg_parser.parse_args()
    src_dirs: Path = [args.source_directory]
    if args.recursive:
        src_dirs = list(filter(lambda x: len(list(x.glob("*.h5"))) > 0, src_dirs[0].glob("**")))
    target_dirs = [folder.with_suffix(".h5") for folder in src_dirs]

    if len(src_dirs) > 1:
        tqdm.set_lock(RLock())
        with ThreadPoolExecutor(
            max_workers=args.procs,
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),)
        ) as pool:
            futures = pool.map(combineHDF5FilesInDirectory, src_dirs, target_dirs)
            list(tqdm(futures, desc="Directory", total=len(src_dirs)))
    elif len(src_dirs) == 1:
        combineHDF5FilesInDirectory(src_dirs[0], target_dirs[0])
    else:
        print("WARNING: No directories with files to convert.")