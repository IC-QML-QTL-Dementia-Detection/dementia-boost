import sys

from dementia_boost.data import OasisDataLoader, OasisDataProcessor


def main() -> None:
    print("--- Testing NIfTI Data Pipeline ---")

    print("\n[1] Running Data Processor...")
    processor = OasisDataProcessor("./data/raw/oasis_longitudinal_demographics.csv")

    processor.process_and_save(
        split_ratio=0.7,
        manual_test_ids=["OAS2_0001", "OAS2_0002"],
    )

    print("\n[2] Initializing DataLoader...")
    loader_manager = OasisDataLoader(batch_size=4)

    try:
        train_loader = loader_manager.get_data_loader(is_train=True)
        print(f"Success! Created DataLoader with {len(train_loader)} batches.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n[3] Inspecting the first batch...")
    for images, labels in train_loader:
        print(f"Images tensor shape: {images.shape}")
        print(f"Images data type:    {images.dtype}")
        print(f"Global Min value:    {images.min().item():.4f}")
        print(f"Global Max value:    {images.max().item():.4f}")
        print(f"Global Mean value:   {images.mean().item():.4f}")

        print(f"\nLabels batch size:   {len(labels)}")
        print(f"Labels data type:    {type(labels)}")
        print(f"Sample Labels:       {labels}")

        break

    print("\nPipeline test complete!")


if __name__ == "__main__":
    main()
