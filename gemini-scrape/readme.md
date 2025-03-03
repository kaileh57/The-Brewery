# S1 Gemini Data Collector

A robust, scalable framework for collecting reasoning traces from Gemini API for the s1 distillation project as described in the paper "s1: Simple test-time scaling".

## Features

- **Multiple API Keys**: Use multiple Gemini API keys simultaneously for maximum throughput
- **Parallel Processing**: Configurable number of concurrent sessions to optimize collection speed
- **Robust Error Handling**: Automatic retries with exponential backoff and jitter
- **Progress Tracking**: Save progress and resume collection from where you left off
- **Flexible Prompting**: Customizable prompts to elicit high-quality reasoning traces
- **Paper-Compatible Output**: Data is formatted according to the s1 paper specifications
- **Secure Configuration**: Uses .env file for sensitive API keys

## Setup

1. **Install dependencies**:
   ```bash
   pip install google-generativeai aiohttp tqdm datasets python-dotenv
   ```

2. **Set up your .env file**:
   ```bash
   # Copy the template
   cp .env.template .env
   
   # Edit with your API keys
   nano .env  # or use any text editor
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   # Create the virtual environment
   python -m venv venv
   
   # Activate it (Windows)
   venv\Scripts\activate
   
   # Activate it (macOS/Linux)
   source venv/bin/activate
   
   # Install dependencies
   pip install google-generativeai aiohttp tqdm datasets python-dotenv
   ```

## .env Template

```
# API Keys - add as many as you have
GEMINI_API_KEY_1=your_api_key_1
GEMINI_API_KEY_2=your_api_key_2
GEMINI_API_KEY_3=your_api_key_3
# For single-key setups, you can also use:
# GEMINI_API_KEY=your_single_api_key

# Model configuration
GEMINI_MODEL_NAME=gemini-2.0-flash-thinking-exp-1219

# Performance settings
MAX_CONCURRENT_SESSIONS=15
REQUESTS_PER_MINUTE=10

# Output configuration
OUTPUT_DIR=./s1_collected_data
```

## Usage

1. **Format your .env file correctly**:
   ```bash
   # Create and edit .env file
   nano .env
   ```
   
   Make sure it has the proper format with line breaks:
   ```
   # API Keys - add as many as you have
   GEMINI_API_KEY_1=your_api_key_1
   GEMINI_API_KEY_2=your_api_key_2
   
   # For single-key setups, you can also use:
   # GEMINI_API_KEY=your_single_api_key
   
   # Model configuration
   GEMINI_MODEL_NAME=gemini-2.0-flash-thinking-exp-1219
   
   # Performance settings
   MAX_CONCURRENT_SESSIONS=15
   REQUESTS_PER_MINUTE=10
   
   # Output configuration
   OUTPUT_DIR=./s1_collected_data
   ```

2. **Verify your .env file**:
   ```bash
   # Check if the .env file is formatted correctly
   cat .env
   ```

3. **Run the collector**:
   ```bash
   python s1_gemini_collector.py --config config.json --env .env
   ```

4. **Output**:
   The script creates an output directory with:
   - `progress.json`: Tracking processed samples
   - `outputs.json`: Complete raw outputs with metadata
   - `s1k_format.json`: Data formatted for s1 distillation (matches paper format)

## Configuration Options

### Collector Config (.env and config.json)

| Option | .env Variable | Description |
|--------|---------------|-------------|
| `api_keys` | `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc. | List of Gemini API keys |
| `model_name` | `GEMINI_MODEL_NAME` | Gemini model name |
| `max_concurrent_sessions` | `MAX_CONCURRENT_SESSIONS` | Maximum concurrent API sessions |
| `requests_per_minute_per_key` | `REQUESTS_PER_MINUTE` | Rate limit per API key |
| `output_dir` | `OUTPUT_DIR` | Directory to save outputs |
| `log_level` | - | Logging level (from config.json) |
| `resume` | - | Whether to resume from previous progress (from config.json) |
| `max_retries` | - | Maximum number of retries for API calls (from config.json) |
| `save_interval` | - | How often to save progress (from config.json) |

### Dataset Config (config.json)

| Option | Description |
|--------|-------------|
| `name` | Dataset name (from Hugging Face) |
| `split` | Dataset split (train, test, validation) |
| `num_samples` | Number of samples to process (optional) |
| `seed` | Random seed for shuffling |
| `question_field` | Field to extract questions from |
| `fallback_fields` | Alternative fields to try if question_field is missing |

### Prompt Config (config.json)

| Option | Description |
|--------|-------------|
| `template` | Template for question formatting |
| `system_prompt` | System prompt for Gemini model |
| `format_instructions` | Instructions for reasoning format |

## Maximizing Throughput

To maximize throughput:

1. **Add more API keys**: Add multiple `GEMINI_API_KEY_X` entries in your .env file
2. **Adjust concurrent sessions**: Update `MAX_CONCURRENT_SESSIONS` in your .env file
3. **Fine-tune rate limits**: Set `REQUESTS_PER_MINUTE` based on your API tier quota

## Paper Compatibility

The collector is designed to produce data in the format used by the s1 paper:
- High-quality reasoning traces from Gemini's thinking API
- Properly separated reasoning and answers
- Structured for direct use in the distillation process
