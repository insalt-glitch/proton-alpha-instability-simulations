#!/home/nilsm/miniconda3/envs/analysis/bin/python3
"""Utility to combine many HDF5-files.
"""
import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from os import process_cpu_count
from functools import partial
from pathlib import Path
from threading import RLock
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

INFO_GROUPS = ["/Run_info", "/Header"]
CONSTANT_GROUPS = ["/Grid/CPUs"] # "/Grid/grid"

# TODO: This script should be able to operate on a number of files as well (instead of folders)
# TODO: We cannot handle changing attributes in general currently (not even on datasets). In that case, we could just create a group with multiple datasets in it.
# It is slightly complicated however to infer which datasets are changing before doing the merge. (We don't have unlimited memory)
# TODO: Would be nice to be able to specify a desired max. File size and than split up the result in smaller files.
# TODO: There is no way to include/exclude subgroups of other groups
# TODO: There is no way to handle extra data on some files (currently crashes).

def isIncluded(
    path: str,
    included_paths: Iterable[str],
    excluded_paths: Iterable[str]
) -> bool:
    path = path.removeprefix("/")
    included_paths = {p.removeprefix("/") for p in included_paths}
    excluded_paths = {p.removeprefix("/") for p in excluded_paths}
    assert included_paths.isdisjoint(excluded_paths), "Cannot include and exclude at the same time"

    is_excluded = any(path.startswith(excl) for excl in excluded_paths)
    is_included_directly = any(path.startswith(incl) for incl in included_paths)

    if len(included_paths) == 0:
        return not is_excluded
    if not is_included_directly:
        return False
    if not is_excluded:
        return True
    nested_excludes = {excl for excl in excluded_paths if not any(incl.startswith(excl) for incl in included_paths)}
    is_excluded = any(path.startswith(excl) for excl in nested_excludes)
    return not is_excluded

""" ---------------------------------------------------------------------------
Initialization
--------------------------------------------------------------------------- """

def createAttributeStore(src_file: h5py.File, num_files: int) -> dict[str, Any]:
    assert num_files > 0, "Number of files must positive."
    # Create a dict with keys Run_info/<attr_key>.
    # Values should be numpy array or list of appropiate size [None] * time.size
    attr_store = {}
    for group_name in INFO_GROUPS:
        assert group_name in src_file, f"Info group '{group_name}' does not exist in '{src_file.filename}'."
        h5_group = src_file[group_name]
        assert len(h5_group.keys()) == 0, f"Info group '{group_name}' in '{src_file.filename}' cannot contain objects."
        for attr_name, attr_value in h5_group.attrs.items():
            # build unique key
            key = f"{group_name}/{attr_name}"
            assert key not in attr_store, f"'{key}' already in attribute store."
            # NOTE: Assume that value is string of numpy-like type.
            if isinstance(attr_value, str):
                attr_store[key] = [None] * num_files
            else:
                assert hasattr(attr_value, "dtype") and hasattr(attr_value, "shape"), f"Expected numpy-like type of '{key}' in '{src_file.filename}'."
                attr_store[key] = np.empty(shape=(num_files, *attr_value.shape), dtype=attr_value.dtype)
    return attr_store

def createTargetDatasets(src_file: h5py.File, target_file: h5py.File, num_files: int) -> None:
    assert target_file.mode == "r+", f"Write permission on target-file '{target_file.filename}' required."
    assert num_files > 0, "Number of files must positive."
    # Recursively go through and create groups/datasets not in 'Header' or 'Run_info' or 'Grid'.
    # Assert that the groups do not have attributes. Copy the attributes from the other datasets
    def _createDatasets(name: str, h5_obj: h5py.Group|h5py.Dataset):
        if any(name.startswith(group_name.removeprefix("/")) for group_name in INFO_GROUPS + CONSTANT_GROUPS):
            return
        if isinstance(h5_obj, h5py.Group):
            assert len(h5_obj.attrs.keys()) == 0, f"Group '{name}' in '{target_file.filename}' is not allowed to have attributes."
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
    assert target_file.mode == "r+", f"Write permission on target-file '{target_file.filename}' required."
    # Copy datasets in 'Grid' to new file as they are
    for group_name in CONSTANT_GROUPS:
        src_file.copy(
            source=group_name,
            dest=target_file,
            name=group_name,
            shallow=False
        )

""" ---------------------------------------------------------------------------
Data transfer
--------------------------------------------------------------------------- """

def copyDatasets(file_idx: int, src_file: h5py.File, target_file: h5py.File):
    assert target_file.mode == "r+", f"Write permission on target-file '{target_file.filename}' required."
    # For each file copy all datasets not in 'Header' or 'Run_info' or 'Grid'
    # assert that the attributes are the same and groups do not have attributes
    # assert that the dimensions are the same
    def _copyData(name: str, h5_obj: h5py.Group|h5py.Dataset):
        # skip special paths
        if any(name.startswith(group_name.removeprefix("/")) for group_name in INFO_GROUPS + CONSTANT_GROUPS):
            return
        # check conformity
        assert name in target_file, f"Object '{name}' in '{src_file.filename}' does not exist on target-file '{target_file.filename}'."
        if isinstance(h5_obj, h5py.Group):
            assert len(h5_obj.attrs.keys()) == 0, f"Group '{name}' in '{src_file.filename}' is not allowed to have attributes."
            return
        src_ds = h5_obj
        target_ds = target_file[name]
        assert src_ds.shape == target_ds.shape[1:], f"Dimensions ({src_ds.shape} vs. {target_ds.shape[1:]} of '{name}' in '{src_file.filename}' do not match."
        # TODO: Temporary work-around because we donÄt know what to do with changin attributes on datasets
        # assert src_ds.attrs == target_ds.attrs, f"Attributes of '{name}' in '{src_file.filename}' do not match."
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
            assert attr_name not in h5_group.attrs, f"Attribute '{attr_name}' already exists in target file '{target_file.filename}'."
            h5_group.attrs[attr_name] = dataset[0]
        else:
            assert name not in target_file, f"Dataset '{name}' already exists in target file '{target_file.filename}'."
            h5_group[attr_name] = dataset

""" ---------------------------------------------------------------------------
Main functions
--------------------------------------------------------------------------- """

def combineHDF5FilesInDirectory(
    src_files: list[Path],
    target_file: Path,
    reference_file: Path
) -> None:
    assert len(src_files) > 0, "No src-iles provided."
    assert all(file.exists() for file in src_files), "All src-files have to exist."
    assert reference_file in src_files, f"Reference file '{reference_file}' has to be part of src_files."
    assert not target_file.exists(), f"Target file '{target_file}' will not be overwritten."

    # initial setup
    num_files = len(src_files)
    with h5py.File(reference_file) as h5_src:
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

""" ---------------------------------------------------------------------------
Command-line stuff
--------------------------------------------------------------------------- """

def _validateSrcPath(arg: str) -> Path:
    """Validates that the argument is an existing folder.

    Args:
        arg (str): folder-path

    Returns:
        Path: valid folder-path
    """
    src_path = Path(arg)
    if not src_path.exists():
        raise argparse.ArgumentTypeError(f"Directory '{src_path}' does not exist")
    return src_path.absolute()

def _validateSrcFile(arg: str) -> Path:
    """Validates that the argument is an existing folder.

    Args:
        arg (str): folder-path

    Returns:
        Path: valid folder-path
    """
    src_path = Path(arg)
    if not src_path.exists():
        raise argparse.ArgumentTypeError(f"Directory '{src_path}' does not exist")
    if not src_path.is_file():
        raise argparse.ArgumentTypeError(f"'{src_path}' is not a file.")
    return src_path.absolute()

def _validateNumProcs(arg: str) -> int:
    if not arg.isdigit():
        raise argparse.ArgumentTypeError("Supply a positive number of processes")
    n_procs = int(arg)
    if n_procs <= 0:
        raise argparse.ArgumentTypeError("Need at least one process")
    return n_procs

def _parseComandlineArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="h5vstack",
        description="Stacks datasets inside HDF5-files along a new dimension. "
            "Attributes not part of groups specifed in INFO_GROUPS are assumed "
            "to be the same (crash). Groups specified in CONSTANT_GROUPS will "
            "only be copied once."
    )
    parser.add_argument(
        "-r", "--recursive",
        help="Recursively convert SDF-files in all sub-directories. Defaults to off.",
        action="store_true"
    )
    parser.add_argument(
        "-p", "--procs",
        help="Number of processors to use. Defauts to the number of CPU-threads.",
        type=_validateNumProcs, default=process_cpu_count(), metavar="N_PROCS>=1"
    )
    parser.add_argument(
        "--reference-file",
        help="File is used as a references for the HDF5-file structure.",
        type=_validateSrcFile
    )
    # TODO: Target file option is broken currently. We need to have target-file when only files are supplied.
    parser.add_argument(
        "--target-file",
        help="Target/Output file.",
        # type=_validateSrcFile
    )
    parser.add_argument_group()
    group = parser.add_mutually_exclusive_group()
    group.title = "Target-file options"
    group.add_argument(
        "--overwrite-target",
        help="Overwrite existing target-files.",
        action="store_true"
    )
    group.add_argument(
        "--skip-existing",
        help="Skip existing target-files.",
        action="store_true"
    )
    parser.add_argument(
        "PATH",
        help="Input files (HDF5) or folders. Cannot be mixed.",
        type=_validateSrcPath, nargs="+"
    )
    args = parser.parse_args()
    only_files = all(p.is_file() for p in args.PATH)
    only_dirs = all(p.is_dir() for p in args.PATH)
    if not (only_files or only_dirs):
        parser.error("You can only specify files OR folders.")
    if only_files and args.recursive:
        parser.error("'--recursive' cannot be used with files")
    if only_dirs and args.target_file is not None:
        parser.error("'--target-file' cannot be used with folders")
    if only_dirs and args.reference_file is not None:
        parser.error("'--reference-file' cannot be used with folders")

    return args

def _prepareFiles(args: argparse.Namespace) -> tuple[list[list[Path]],list[Path],list[Path]]:
    # Handle single files
    if all(p.is_file() for p in args.PATH):
        src_files: list[Path] = sorted(args.PATH)
        if args.target_file is None:
            target_file: Path = src_files[-1]
            src_files = src_files[:-1]
        else:
            target_file: Path = args.target_file
        assert not target_file.exists(), f"Target file '{target_file.as_posix()}' exists."

        if args.reference_file is None:
            reference_file = src_files[0]
        else:
            reference_file: Path = args.reference_file
        return [src_files], [target_file], [reference_file]
    # Handle directories
    if args.recursive:
        src_dirs: list[Path] = sorted(
            sub_folder for folder in args.PATH for sub_folder in folder.glob("**")
        )
    else:
        src_dirs: list[Path] = args.PATH
    src_files: list[list[Path]] = [
        sorted(folder.glob("*.h5")) for folder in set(src_dirs) if len(list(folder.glob("*.h5"))) > 1
    ]
    target_files = [
        file_batch[0].parent / f"{file_batch[0].parent.name}.h5"
        for file_batch in src_files
    ]
    # check conformity with arguments
    if any(file.exists() for file in target_files):
        existing_target_files = [file for file in target_files if file.exists()]
        if args.skip_existing:
            # Filter src_dirs and target_files for existing files
            src_files = [files for files, t in zip(src_files, target_files) if not t.exists()]
            target_files = [t for t in target_files if not t.exists()]
        elif args.overwrite_target:
            # Remove exiting files
            for t in existing_target_files: t.unlink()
        else:
            raise FileExistsError(
                f"Overwrite (--overwrite-target) or skip (--skip-existing) files:\n"
                f"{'\n'.join([str(f) for f in existing_target_files])}")
    return src_files, target_files, [files[0] for files in src_files]

if __name__ == "__main__":
    args = _parseComandlineArgs()
    src_folders, target_files, reference_files = _prepareFiles(args)
    assert len(src_folders) == len(target_files)
    assert len(src_folders) == len(reference_files)

    if len(src_folders) > 1:
        tqdm.set_lock(RLock())
        with ThreadPoolExecutor(
            max_workers=args.procs,
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),)
        ) as pool:
            futures = pool.map(
                combineHDF5FilesInDirectory,
                src_folders,
                target_files,
                reference_files,
            )
            list(tqdm(futures, desc="Directory", total=len(src_folders)))
    elif len(src_folders) == 1:
        combineHDF5FilesInDirectory(src_folders[0], target_files[0], reference_files[0])
    else:
        print("INFO: No directories with files to convert.")
