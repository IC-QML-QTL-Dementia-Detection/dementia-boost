"""ETL pipeline entry-point for raw NIfTI/HDR OASIS-2 MRI volumes.

This script executes the offline ETL pipeline for 3D NIfTI/HDR brain MRI volumes,
performing deterministic patient-level splitting into train, val, and test
cohorts, extracting the central 2D axial slice, serializing processed tensors to
disk with a `split_manifest.json`, and validating the resulting PyTorch
DataLoaders. It is safe to rerun: the `.pt` files of the cohort directories are
replaced, and the same arguments give the same split.
"""

import sys

from dementia_boost.data import OasisDataLoader, OasisDataProcessor
from dementia_boost.data.split_guards import SplitIntegrityError
from dementia_boost.data.split_manifest import read_manifest
from dementia_boost.telemetry.logger import setup_logger

CSV_PATH = "./data/raw/oasis_longitudinal_demographics.csv"
SEED = 42
TEST_RATIO = 0.3
VAL_RATIO = 0.2
# Subjects with most of the images of their class, kept in train as in the paper.
MANUAL_TRAIN_IDS = ["OAS2_0001", "OAS2_0002"]
MANUAL_TEST_IDS: list[str] = []


def main() -> None:
    """Executes the NIfTI ETL pipeline and verifies the cohort DataLoaders."""
    logger = setup_logger("nifti_etl_pipeline")

    logger.info("Initializing NIfTI Data Processor...")
    processor = OasisDataProcessor(csv_path=CSV_PATH)

    processor.process_and_save(
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        manual_train_ids=MANUAL_TRAIN_IDS,
        manual_test_ids=MANUAL_TEST_IDS,
    )
    manifest = read_manifest(processor.manifest_path)
    logger.info(
        f"Split manifest written to {processor.manifest_path} "
        f"(split_id {manifest['split_id']})"
    )
    logger.info(f"Excluded subjects: {len(processor.excluded)} (see the manifest)")

    logger.info("Initializing NIfTI DataLoader factory...")
    loader_manager = OasisDataLoader(batch_size=64)

    try:
        loaders = {
            cohort: loader_manager.get_data_loader(cohort)
            for cohort in ("train", "val", "test")
        }
    except (FileNotFoundError, SplitIntegrityError) as error:
        logger.error(f"Failed to load dataset: {error}")
        sys.exit(1)
    logger.info(
        "Successfully created DataLoaders: "
        + ", ".join(f"{len(loader)} {name} batches" for name, loader in loaders.items())
    )

    logger.info("Inspecting sample training batch...")
    for images, labels in loaders["train"]:
        logger.info(f"Images tensor shape: {images.shape}")
        logger.info(f"Images data type:    {images.dtype}")
        logger.info(f"Global Min value:    {images.min().item():.4f}")
        logger.info(f"Global Max value:    {images.max().item():.4f}")
        logger.info(f"Global Mean value:   {images.mean().item():.4f}")
        logger.info(f"Labels batch size:   {len(labels)}")
        logger.info(f"Labels data type:    {type(labels)}")
        logger.info(f"Sample Labels:       {labels}")
        break

    logger.info("NIfTI ETL pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
