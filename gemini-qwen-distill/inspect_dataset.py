from datasets import load_dataset

# Load a small sample of the dataset
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.select(range(3))  # Just look at 3 examples

# Print the column names
print("Dataset column names:", dataset.column_names)

# Print the first example
print("\nFirst example type:", type(dataset[0]))
print("First example:", dataset[0])

# Try a different approach - get the first example as dict
example_dict = dataset[0]
print("\nFirst example as dict keys:", example_dict.keys())

# Print the raw features
print("\nDataset Features:")
print(dataset.features)

# Try accessing specific columns
if 'instruction' in dataset.column_names:
    print("\nSample instruction:", dataset[0]['instruction'][:100], "...")
    
if 'output' in dataset.column_names:
    print("\nSample output:", dataset[0]['output'][:100], "...")

# Print a complete sample in detail
print("\nDetailed sample:")
for i, sample in enumerate(dataset):
    print(f"\nSample {i}:")
    for k, v in sample.items():
        print(f"  {k}: {v[:100] if isinstance(v, str) else v}...")
