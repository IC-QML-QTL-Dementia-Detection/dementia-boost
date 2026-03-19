import sys

from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.data.data_procesor import OasisDataProcessor


def main() -> None:
    print("--- Testing HDR Data Pipeline ---")

    # ---------------------------------------------------------
    # Step 1: Run the ETL process (Extract, Transform, Load)
    # ---------------------------------------------------------
    print("\n[1] Running Data Processor...")
    processor = OasisDataProcessor()

    processor.process_and_save(split_ratio=0.8)

    # ---------------------------------------------------------
    # Step 2: Initialize the DataLoader
    # ---------------------------------------------------------
    print("\n[2] Initializing DataLoader...")
    loader_manager = OasisDataLoader(batch_size=4)

    try:
        train_loader = loader_manager.get_data_loader(is_train=True)
        print(f"Success! Created DataLoader with {len(train_loader)} batches.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 3: Inspect a single batch
    # ---------------------------------------------------------
    print("\n[3] Inspecting the first batch...")
    for images, labels in train_loader:
        # Expected: [Batch, Channels, Height, Width]
        print(f"Images tensor shape: {images.shape}")
        # Expected: torch.float32
        print(f"Images data type:    {images.dtype}")

        # Key HDR metrics - checking if values exceed standard 1.0 bounds
        print(f"Global Min value:    {images.min().item():.4f}")
        print(f"Global Max value:    {images.max().item():.4f}")
        print(f"Global Mean value:   {images.mean().item():.4f}")

        print(f"\nLabels tensor shape: {labels.shape}")
        print(f"Labels data type:    {labels.dtype}")
        print(f"Sample Labels:       {labels.tolist()}")

        break

    print("\nPipeline test complete!")


if __name__ == "__main__":
    main()
