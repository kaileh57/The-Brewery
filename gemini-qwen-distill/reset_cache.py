import os
import json
from pathlib import Path

# Default cache directory
cache_dir = Path("./gemini_cache")

# Check if the directory exists
if not cache_dir.exists():
    print(f"Cache directory {cache_dir} doesn't exist. Nothing to reset.")
    exit(0)
    
# Files to delete or reset
progress_file = cache_dir / "progress.json"
cache_file = cache_dir / "gemini_responses.json"

# Check and handle the progress file
if progress_file.exists():
    print(f"Deleting progress file: {progress_file}")
    os.remove(progress_file)
else:
    print(f"Progress file {progress_file} not found.")

# Check and handle the cache file
if cache_file.exists():
    # Option 1: Delete the cache file
    # os.remove(cache_file)
    # print(f"Deleted cache file: {cache_file}")
    
    # Option 2: Reset the cache file to an empty array
    with open(cache_file, 'w') as f:
        json.dump([], f)
    print(f"Reset cache file to empty array: {cache_file}")
else:
    print(f"Cache file {cache_file} not found.")
    
print("\nCache reset complete. You can now run gemini_distill.py to start processing from the beginning.")
