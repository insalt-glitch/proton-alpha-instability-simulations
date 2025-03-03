#!/home/nilsm/miniconda3/envs/analysis/bin/python3
"""Utility to convert SDF-files to HDF5-format.
"""
from pathlib import Path
import argparse
import tarfile
from typing import List, Any

import h5py
from tqdm import tqdm
import sdf_helper as sdf
from sdf_helper.sdf import BlockPlainVariable, BlockPlainMesh, BlockConstant

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
    block: BlockPlainVariable|BlockPlainMesh|BlockConstant
) -> None:
    """Save a block from the SDF-format to the HDF5-file.

    Args:
        h5_file (h5py.File): HDF5-file handle
        block (BlockPlainVariable|BlockPlainMesh|BlockConstant): SDF-block that should be saved
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
            dataset = h5_obj.create_dataset(name=axis_label, data=block_data[idx])
            for attr_name in ["extents", "mult", "units"]:
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

def saveSDFFileToHDF5(sdf_file_path: Path, h5_file_path: Path) -> None:
    """Save an SDF-file to HDF5-format.   

    Args:
        sdf_file_path (Path): filepath to the SDF-file
        h5_file_path (Path): filepath to the HDF5-file
    """
    assert sdf_file_path.exists(), "File should exist"
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
                saveNativeBlock(h5_file, block)

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
        for file_path in tqdm(sdf_files, desc="TAR-archive", leave=False):
            tar.add(name=file_path, arcname=file_path.name, recursive=False)

def processDirectory(
    data_directory: Path,
    hdf5_subdirectory: str="",
    archive_name: str="original_sdf-files.tar.xz",
    overwrite: bool=False
) -> None:
    """Processes a directory of SDF-files and covert them to HDF5-format.
    The old files are saved to a tar-archive and then delete.

    Args:
        data_directory (Path): directory that contains the SDF-files
        hdf5_subdirectory (str, optional): If the HDF5-files should be created in a sub-directory.
            Defaults to "".
        archive_name (str, optional): Name of the tar-archive. Defaults to
            "original_sdf-files.tar.xz".
        overwrite (bool, optional): Whether to overwrite any existing files. Defaults to False.
            This skips thexisting files.
    """
    output_folder = data_directory / hdf5_subdirectory
    output_folder.mkdir(exist_ok=True, parents=False)
    sdf_files = sorted(output_folder.glob("*.sdf", case_sensitive=False))
    # convert all SDF files in directory to HDF5
    for sdf_file_path in tqdm(sdf_files, desc="SDF -> HDF5", leave=False):
        h5_file_path = (output_folder / sdf_file_path.name).with_suffix(".h5")
        if h5_file_path.exists() and not overwrite:
            continue
        saveSDFFileToHDF5(sdf_file_path, h5_file_path)

    archive_path = data_directory / archive_name
    if not archive_path.exists() or overwrite:
        archiveSDFFiles(sdf_files, archive_path)
        for sdf_file_path in sdf_files:
            sdf_file_path.unlink()

def validatePathArgument(arg: str) -> Path:
    """Validates the argument of the given path.

    Args:
        arg (str): filepath

    Returns:
        Path: valid file path
    """
    path = Path(arg)
    if not path.exists():
        raise argparse.ArgumentError("Specified directory does not exist")
    if not path.is_dir():
        raise argparse.ArgumentError("Specified path is not a directory")
    return path

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-r", "--recursive",
        help="Recursively convert SDF-files in all sub-directories",
        action="store_true")
    arg_parser.add_argument(
        "--overwrite",
        help="If set will overwrite existing files/archives otherwise these wil be skipped",
        action="store_true"
    )
    arg_parser.add_argument(
        "directory",
        help="Location of the simulation-data",
        type=validatePathArgument
    )
    args = arg_parser.parse_args()

    if args.recursive:
        directories = sorted(args.directory.glob("**"))
    else:
        directories = [args.directory]
    directories = list(filter(lambda x: len(list(x.glob("*.sdf"))) > 0, directories))
    if len(directories) == 0:
        print("WARNING: No SDF-files found")
    else:
        for folder in tqdm(directories, desc="Directory"):
            processDirectory(folder, overwrite=args.overwrite)
