import os
from gemini_wrapper import GeminiTeacherWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
config = {
    "project_name": "gemini-to-qwen-distil",
    "dataset": {
        "name": "mlabonne/FineTome-100k",  # Or another suitable dataset
        "split": "train",
        "num_samples": 5000,  # Start with a smaller subset
        "seed": 42
    },
    "models": {
        "teacher": "gemini-wrapper",  # We'll handle this specially
        "student": "Qwen/Qwen2-7B"  # Choose an appropriate Qwen model size
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results-gemini-qwen",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,
        "fp16": False,
        "bf16": True
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": True
    }
}

# Initialize the teacher and student models
def get_models(config):
    # Initialize the Gemini wrapper with your API key
    teacher = GeminiTeacherWrapper(
        api_key=os.environ.get("GEMINI_API_KEY", "your_api_key_here"),
        tokenizer_name=config["models"]["student"]  # Use the student's tokenizer
    )
    
    # Initialize the student model
    student = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float32,
        use_flash_attention_2=config["model_config"]["use_flash_attention"],
    )
    
    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"],
        padding_side="right",
        use_fast=False,
    )
    
    return teacher, student, tokenizer

# Continue with the rest of your distillation script...