# Gemini-Qwen Chain-of-Thought Distillation

This project implements a simplified approach to distill reasoning capabilities from the Gemini Flash Thinking model to Qwen 1.5B. Using prompt engineering, we encourage Gemini to produce chain-of-thought responses, which are then used to train Qwen to develop similar reasoning abilities.

## Overview

The project consists of three main components:

1. **API Testing**: A simple script to verify your Gemini API connection works correctly
2. **Data Collection**: A script that generates reasoning-oriented questions and collects chain-of-thought responses from Gemini
3. **Model Training**: A script that trains the Qwen model on these reasoning examples

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (for training)
- Gemini API key (obtain from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Setup Steps

1. **Clone the repository or create a new directory**:
   ```
   mkdir gemini-qwen-distill
   cd gemini-qwen-distill
   ```

2. **Create a virtual environment**:
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source venv/bin/activate
     ```

4. **Install PyTorch**:
   ```
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Install other dependencies**:
   ```
   pip install transformers>=4.35.0 trl>=0.7.2 accelerate>=0.21.0 datasets>=2.14.0 python-dotenv>=1.0.0 google-generativeai>=0.3.0
   ```

   > **Note**: We recommend NOT installing flash-attention as it can cause compatibility issues on Windows and with certain GPUs. The scripts are configured to work without it by default.

6. **Create the necessary files**:

   Create `test_gemini_api.py`, `simple_gemini_distill.py`, and `train_qwen.py` using the provided code.

7. **Create a `.env` file** with your configuration:
   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL_NAME=gemini-2.0-flash-thinking-exp-1219
   STUDENT_MODEL=Qwen/Qwen2-1.5B
   NUM_SAMPLES=15
   CACHE_DIR=./gemini_cache
   OUTPUT_DIR=./gemini-qwen-results
   MAX_LENGTH=4096
   NUM_EPOCHS=3
   BATCH_SIZE=1
   LEARNING_RATE=2e-5
   # Default is flash attention OFF
   USE_FLASH_ATTENTION=False
   ```

## Usage

### Step 1: Test the API Connection

Before generating examples, verify your Gemini API connection:

```
python test_gemini_api.py
```

This script will attempt to connect to the Gemini API and display a sample response. If successful, you'll see a confirmation message.

### Step 2: Generate Chain-of-Thought Responses

Run the data collection script to generate reasoning examples:

```
python simple_gemini_distill.py
```

This script:
- Creates reasoning-oriented prompts that require step-by-step thinking
- Explicitly asks Gemini to show its reasoning process
- Respects the Gemini API rate limit (10 calls/minute)
- Saves responses to `gemini_cache/conversations.json`
- Displays progress as it works

The generation process will be slow due to rate limits. For 15 examples, expect it to take at least 15-20 minutes.

### Step 3: Train the Qwen Model

Once you've generated examples, train the Qwen model:

```
python train_qwen.py
```

This script:
- Loads the conversation examples
- Creates a training dataset with proper formatting
- Fine-tunes the Qwen 1.5B model
- Saves the trained model to the output directory

Training time depends on your GPU. On a typical consumer GPU, expect several hours for a complete training run.

## Customization

### Adjusting the Number of Examples

To change how many examples are generated, update the `NUM_SAMPLES` value in your `.env` file.

### Customizing Reasoning Tasks

To focus on specific types of reasoning, edit the `topics` list in `simple_gemini_distill.py`.

### Training Configuration

You can adjust training parameters in the `.env` file:
- `NUM_EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate
- `MAX_LENGTH`: Maximum sequence length

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors during training:
1. Reduce `BATCH_SIZE` in your `.env` file
2. Reduce `MAX_LENGTH` to process shorter sequences
3. Use gradient accumulation by increasing the `GRAD_ACCUM_STEPS` parameter

### API Rate Limits

The Gemini API has strict rate limits. If you encounter rate limit errors:
1. The script will automatically retry with exponential backoff
2. You can increase `DELAY_BETWEEN_CALLS` in `simple_gemini_distill.py` if needed

### Windows-Specific Issues

If you encounter access violations or memory errors on Windows:
1. Ensure you're running the latest version of PyTorch
2. Keep `USE_FLASH_ATTENTION=False` in your `.env` file
3. Try running the scripts with lower batch sizes

## Why No Flash Attention?

We recommend against using Flash Attention because:
1. It requires compilation from source, which can be difficult on Windows
2. It may not be compatible with all GPU architectures
3. It can cause memory access violations on some systems
4. The performance benefit isn't necessary for this specific task

The scripts will work fine without Flash Attention, as they're configured to use standard attention mechanisms by default.

## Advanced: Using With Flash Attention (Not Recommended)

If you still want to try Flash Attention:

1. Set `USE_FLASH_ATTENTION=True` in your `.env` file
2. Install Flash Attention (may require build tools):
   ```
   pip install packaging
   pip install flash-attn --no-build-isolation
   ```

Note that this may cause stability issues, especially on Windows.
