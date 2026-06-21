from dementia_boost.data.jpg_indexer import JpgDataIndexer


def main() -> None:
    raw_jpg_input = "./data/raw/jpg"
    index_output_destination = "./data/results/jpg"

    print("Initializing Offline JPG Dataset Indexer...")
    indexer = JpgDataIndexer(
        raw_jpg_dir=raw_jpg_input,
        output_dir=index_output_destination,
    )

    indexer.index_and_split(
        split_ratio=0.7,
        seed=158,
        manual_train_ids=[],
        manual_test_ids=[],
    )


if __name__ == "__main__":
    main()
