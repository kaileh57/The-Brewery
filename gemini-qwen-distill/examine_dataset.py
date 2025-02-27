from datasets import load_dataset

# Load a small sample of the dataset
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.select(range(3))  # Just look at 3 examples

# Print the available keys in the dataset
print("Dataset keys:", dataset[0].keys())

# Print a complete example
print("\nExample 0:")
print(dataset[0])

# If there's a nested structure, let's examine it in detail
print("\nDetailed structure:")
for key, value in dataset[0].items():
    print(f"\n{key}:")
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"  {k}: {v}")
    elif isinstance(value, list):
        print(f"  List with {len(value)} items")
        for i, item in enumerate(value[:2]):  # Show first 2 items
            print(f"  {i}: {item}")
        if len(value) > 2:
            print(f"  ... {len(value)-2} more items")
    else:
        print(f"  {value}")
