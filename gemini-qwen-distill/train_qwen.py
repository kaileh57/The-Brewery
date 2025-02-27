"""
Training script for Qwen model based on generated Gemini responses
"""
import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
STUDENT_MODEL = os.getenv("STUDENT_MODEL", "Qwen/Qwen2-1.5B")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "4096"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./gemini-qwen-results")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./gemini_cache"))
responses_file = CACHE_DIR / "conversations.json"

def main():
    # Check if conversations file exists
    if not responses_file.exists():
        print(f"Error: Conversations file '{responses_file}' not found!")
        print("Please run simple_gemini_distill.py first to generate responses.")
        return
        
    # Load the conversations
    with open(responses_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations for training")
    if len(conversations) == 0:
        print("No conversations found - cannot train model")
        return
        
    # Load tokenizer
    print(f"Loading tokenizer: {STUDENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure we have a chat template
    if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        
    # Create training examples
    training_examples = []
    for conv in conversations:
        # Format as a conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": conv["question"]},
            {"role": "assistant", "content": conv["response"]}
        ]
            
        # Apply the chat template
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        training_examples.append({"text": formatted_text})
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(training_examples)
    print(f"Created dataset with {len(dataset)} examples")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
        
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Split into train and test
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    print(f"Training set: {len(tokenized_dataset['train'])} examples")
    print(f"Test set: {len(tokenized_dataset['test'])} examples")
    
    # Load the model
    print(f"Loading student model: {STUDENT_MODEL}")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention")
    except ImportError:
        print("Flash Attention not available, using default attention implementation")
        
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, **model_kwargs)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=LEARNING_RATE,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        save_steps=50,
        logging_steps=10,
        bf16=True,
        remove_unused_columns=True,
    )
    
    # Create SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=MAX_LENGTH,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    
if __name__ == "__main__":
    main()
