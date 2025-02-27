## Note on Implementation

This implementation uses supervised fine-tuning rather than true logit-based distillation since we can't access the logits or hidden states from the Gemini API. While this approach differs from the original DistillKit methodology, it achieves a similar goal of transferring knowledge from a larger teacher model to a smaller student model.

## Rate Limiting

The Gemini API has a rate limit of 10 calls per minute for the `gemini-2.0-flash-thinking-exp-01-21` model. The script is designed to respect this limit by:

1. Adding a 6-second delay between API calls
2. Implementing retry logic with exponential backoff
3. Saving progress after each processed example
4. Creating a cache of Gemini responses to avoid redundant API calls

This means that generating the training dataset will take considerable time. For example, processing 1,000 examples will take at least 100 minutes due to rate limiting. Consider starting with a small `NUM_SAMPLES` value (e.g., 50-100) to test the pipeline before scaling up.