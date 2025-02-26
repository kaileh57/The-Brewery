"""
Configuration file for Gemini to Qwen distillation
Uses environment variables from .env file for sensitive information
"""
import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Check if required environment variables are set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in .env file. You'll need to provide it to use the Gemini API.")

# Main configuration
config = {
    "project_name": os.getenv("PROJECT_NAME", "gemini-qwen-distillation"),
    
    # Gemini API settings
    "gemini": {
        "api_key": GEMINI_API_KEY,  # From .env file
        "model_name": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-thinking-exp-1219"),
        "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "4096")),
    },
    
    # Student model (Qwen)
    "student": {
        "model_name": os.getenv("STUDENT_MODEL", "Qwen/Qwen2-1.5B"),  # You can choose different sizes: 1.5B, 7B, 14B, etc.
        "use_flash_attention": os.getenv("USE_FLASH_ATTENTION", "False").lower() == "true",
    },
    
    # Dataset settings
    "dataset": {
        "name": os.getenv("DATASET_NAME", "mlabonne/FineTome-100k"),
        "split": os.getenv("DATASET_SPLIT", "train"),
        "num_samples": int(os.getenv("NUM_SAMPLES", "1000")),  # Limit samples for faster training/testing
        "seed": int(os.getenv("SEED", "42")),
        "system_prompt": os.getenv("SYSTEM_PROMPT", "You are a helpful assistant that provides accurate and thoughtful answers.")
    },
    
    # Tokenizer settings
    "tokenizer": {
        "max_length": int(os.getenv("MAX_LENGTH", "2048")),  # Reduced context length to save memory
        "chat_template": os.getenv("CHAT_TEMPLATE", "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
    },
    
    # Training settings
    "training": {
        "output_dir": os.getenv("OUTPUT_DIR", "./results/gemini-qwen-distilled"),
        "num_train_epochs": int(os.getenv("NUM_EPOCHS", "3")),
        "per_device_train_batch_size": int(os.getenv("BATCH_SIZE", "1")),
        "gradient_accumulation_steps": int(os.getenv("GRAD_ACCUM_STEPS", "32")),
        "save_steps": int(os.getenv("SAVE_STEPS", "500")),
        "logging_steps": int(os.getenv("LOGGING_STEPS", "50")),
        "learning_rate": float(os.getenv("LEARNING_RATE", "2e-5")),
        "weight_decay": float(os.getenv("WEIGHT_DECAY", "0.01")),
        "warmup_ratio": float(os.getenv("WARMUP_RATIO", "0.1")),
        "lr_scheduler_type": os.getenv("SCHEDULER", "cosine"),
        "fp16": os.getenv("FP16", "False").lower() == "true",
        "bf16": os.getenv("BF16", "True").lower() == "true",
        "max_grad_norm": float(os.getenv("MAX_GRAD_NORM", "1.0")),
        "save_total_limit": int(os.getenv("SAVE_TOTAL_LIMIT", "3")),
        "group_by_length": os.getenv("GROUP_BY_LENGTH", "True").lower() == "true",
        "use_4bit": os.getenv("USE_4BIT", "True").lower() == "true",
        "use_lora": os.getenv("USE_LORA", "True").lower() == "true",
        "lora_r": int(os.getenv("LORA_R", "16")),
        "lora_alpha": int(os.getenv("LORA_ALPHA", "32")),
        "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.05"))
    },
    
    # Distillation settings
    "distillation": {
        "type": os.getenv("DISTILL_TYPE", "response"),  # Can be "response" or "thinking"
        "temperature": float(os.getenv("TEMPERATURE", "2.0")),
        "alpha": float(os.getenv("DISTILL_ALPHA", "0.7")),  # Weight of distillation loss vs regular loss
        "cache_dir": os.getenv("CACHE_DIR", "./gemini_cache"),  # To avoid repeated API calls
        "max_api_retries": int(os.getenv("MAX_API_RETRIES", "3")),
        "api_timeout": int(os.getenv("API_TIMEOUT", "30")),  # seconds
    }
}