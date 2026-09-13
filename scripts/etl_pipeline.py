"""ETL pipeline entry-point for raw NIfTI/HDR OASIS-2 MRI volumes.

This script executes the offline ETL pipeline for 3D NIfTI/HDR brain MRI volumes,
performing deterministic patient-level cohort splitting, extracting the central
2D axial slice, serializing processed tensors to disk, and validating the resulting
PyTorch DataLoader.
"""

import sys

from dementia_boost.core.reproducibility import set_seed
from dementia_boost.data import OasisDataLoader, OasisDataProcessor
from dementia_boost.telemetry.logger import setup_logger


def main() -> None:
    """Executes the NIfTI ETL pipeline and verifies tensor DataLoader batching."""
    fixed_seed = 42
    set_seed(fixed_seed)
    logger = setup_logger("nifti_etl_pipeline")

    logger.info("Initializing NIfTI Data Processor...")
    processor = OasisDataProcessor(
        csv_path="./data/raw/oasis_longitudinal_demographics.csv",
    )

    processor.process_and_save(
        split_ratio=0.7,
        seed=fixed_seed,
        manual_train_ids=["OAS2_0001", "OAS2_0002"],
        manual_test_ids=[],
    )

    logger.info("Initializing NIfTI DataLoader factory...")
    loader_manager = OasisDataLoader(batch_size=64, mode="nifti")

    try:
        train_loader = loader_manager.get_data_loader(is_train=True)
        test_loader = loader_manager.get_data_loader(is_train=False)
        logger.info(
            f"Successfully created DataLoaders: {len(train_loader)} training batches, "
            f"{len(test_loader)} test batches."
        )
    except FileNotFoundError as error:
        logger.error(f"Failed to load dataset: {error}")
        sys.exit(1)

    logger.info("Inspecting sample training batch...")
    for images, labels in train_loader:
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
