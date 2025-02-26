"""
Data preparation script for distillation from Gemini to Qwen.
"""

import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from typing import Dict, List, Optional
from tqdm.auto import tqdm
import os
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_sharegpt_conversations(example: Dict) -> Dict:
    """
    Format the ShareGPT conversation format into a standard messages format.
    """
    conversations = example['conversations']
    messages = []
    
    # Check if we should add a system message
    has_system_msg = False
    
    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                if conversation.get('from') == 'system':
                    messages.append({"role": "system", "content": conversation.get('value', '')})
                    has_system_msg = True
                elif conversation.get('from') == 'human':
                    messages.append({"role": "user", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'gpt':
                    messages.append({"role": "assistant", "content": conversation.get('value', '')})
    
    # Add default system message if none exists
    if not has_system_msg:
        messages.insert(0, {"role": "system", "content": config["dataset"]["system_prompt"]})
    
    return {"messages": messages}

def prepare_dataset():
    """
    Load and prepare the dataset for distillation.
    """
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    
    # Load dataset
    dataset = load_dataset(
        config["dataset"]["name"], 
        split=config["dataset"]["split"]
    )
    
    # Limit number of samples if specified
    if config["dataset"].get("num_samples"):
        dataset = dataset.select(range(min(len(dataset), config["dataset"]["num_samples"])))
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    
    # Format conversations
    logger.info("Formatting conversations...")
    dataset = dataset.map(format_sharegpt_conversations, remove_columns=dataset.column_names)
    
    # Load student tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(config["student"]["model_name"])
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]
    
    def tokenize_function(examples):
        # Apply the chat template to format the conversation
        formatted_inputs = []
        for messages in examples["messages"]:
            chat_text = student_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_inputs.append(chat_text)
        
        # Tokenize the formatted conversations
        tokenized = student_tokenizer(
            formatted_inputs,
            padding="max_length",
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            return_tensors="pt"
        )
        
        # Prepare labels for autoregressive training
        labels = tokenized["input_ids"].clone()
        
        # Find positions where assistant responses start
        for i, messages in enumerate(examples["messages"]):
            assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]
            if not assistant_msgs:
                continue
                
            # For each example, find where the final assistant message starts
            # We only want to calculate loss on the assistant's responses
            tokens = tokenized["input_ids"][i].tolist()
            text = formatted_inputs[i]
            
            # Get the last assistant message
            last_assistant_msg = assistant_msgs[-1]["content"]
            
            # Find where the last assistant message starts in the tokenized input
            # This is a simple approximation - in real code we would be more precise
            assistant_start_text = "<|im_start|>assistant\n"
            if assistant_start_text in text:
                # Get the position of the last occurrence
                last_pos = text.rfind(assistant_start_text)
                
                # Convert the text position to token index (approximate)
                prefix = text[:last_pos]
                prefix_tokens = student_tokenizer(prefix, add_special_tokens=False)["input_ids"]
                assistant_start_idx = len(prefix_tokens)
                
                # Set labels to -100 for all tokens except the assistant's response
                labels[i, :assistant_start_idx] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names
    )
    
    # Split dataset
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=config["dataset"]["seed"])
    
    logger.info(f"Dataset prepared: {len(tokenized_dataset['train'])} training examples, "
                f"{len(tokenized_dataset['test'])} validation examples")
    
    return tokenized_dataset
    
if __name__ == "__main__":
    dataset = prepare_dataset()
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")
    
    # Show sample
    sample = dataset['train'][0]
    tokenizer = AutoTokenizer.from_pretrained(config["student"]["model_name"])
    
    print("\nSample input:")
    print(tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
    
    # Show where we're actually computing loss (non -100 values)
    print("\nLabels with loss (non -100 values):")
    labels = sample['labels']
    
    # Convert labels to a list if it's not already
    if not isinstance(labels, list):
        try:
            # Try to convert tensor to list
            labels_list = labels.tolist()
        except AttributeError:
            # If it's a simple int, make it a single-item list
            labels_list = [labels]
    else:
        labels_list = labels
        
    # Filter out -100 values
    valid_label_ids = [id for id in labels_list if id != -100 and id >= 0]
    
    if valid_label_ids:
        print(tokenizer.decode(valid_label_ids, skip_special_tokens=True))
    else:
        print("No valid label IDs found")