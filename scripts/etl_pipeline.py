import sys

from dementia_boost.core.reproducibility import set_seed
from dementia_boost.data import OasisDataLoader, OasisDataProcessor
from dementia_boost.telemetry.logger import setup_logger


def main() -> None:
    set_seed(42)
    logger = setup_logger("etl_pipeline")

    logger.info("Testing NIfTI Data Pipeline")

    print()
    logger.info("Running Data Processor...")
    processor = OasisDataProcessor("./data/raw/oasis_longitudinal_demographics.csv")

    processor.process_and_save(
        split_ratio=0.7,
        manual_test_ids=["OAS2_0001", "OAS2_0002"],
    )

    logger.info("Initializing DataLoader...")
    loader_manager = OasisDataLoader(batch_size=4)

    try:
        train_loader = loader_manager.get_data_loader(is_train=True)
        logger.info(f"Success! Created DataLoader with {len(train_loader)} batches.")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    logger.info("Inspecting the first batch...")
    for images, labels in train_loader:
        logger.info(f"Images tensor shape: {images.shape}")
        logger.info(f"Images data type:    {images.dtype}")
        logger.info(f"Global Min value:    {images.min().item():.4f}")
        logger.info(f"Global Max value:    {images.max().item():.4f}")
        logger.info(f"Global Mean value:   {images.mean().item():.4f}")

        logger.info(f"\nLabels batch size:   {len(labels)}")
        logger.info(f"Labels data type:    {type(labels)}")
        logger.info(f"Sample Labels:       {labels}")

        break

    print()
    logger.info("Pipeline test complete!")


if __name__ == "__main__":
    main()
