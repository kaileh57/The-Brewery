"""
Main training script for distilling knowledge from Gemini API to Qwen model.
"""

import os
import torch
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    HfArgumentParser, 
    default_data_collator
)
from data_preparation import prepare_dataset
from gemini_wrapper import GeminiTeacherWrapper, GeminiConfig
from distill_trainer import GeminiDistillationTrainer
from peft import LoraConfig, get_peft_model
from config import config
from accelerate import Accelerator
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('distillation_training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Qwen model by distilling from Gemini API")
    parser.add_argument(
        "--use_lora", 
        action="store_true", 
        help="Whether to use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16, 
        help="Rank of the LoRA matrices"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=float, 
        default=32.0, 
        help="Scaling factor for LoRA"
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=None, 
        help="Gemini API key (can also be set in config.py)"
    )
    parser.add_argument(
        "--use_4bit", 
        action="store_true", 
        help="Whether to use 4-bit quantization for the student model"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None, 
        help="Path to a checkpoint to resume training from"
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set up Accelerate
    accelerator = Accelerator()
    
    # Set API key from args or config
    api_key = args.api_key or config["gemini"]["api_key"]
    if not api_key:
        raise ValueError("Gemini API key must be provided either in config.py or via --api_key")
    
    # Update config with arguments
    if args.resume_from_checkpoint:
        config["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    
    # Prepare the dataset
    logger.info("Preparing dataset...")
    dataset = prepare_dataset()
    
    # Load the tokenizer
    logger.info(f"Loading tokenizer for {config['student']['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["student"]["model_name"])
    tokenizer.chat_template = config["tokenizer"]["chat_template"]
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if config["training"]["bf16"] else torch.float16,
        "device_map": "cuda:0",  # Explicitly use CUDA GPU
    }
    
    # Add flash attention only if configured and available
    if config["student"]["use_flash_attention"]:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 will be used for faster training.")
        except ImportError:
            logger.warning("Flash Attention 2 is enabled in config but not installed. Continuing without it.")
            logger.warning("To install Flash Attention 2, run: pip install flash-attn")
    
    if args.use_4bit:
        model_kwargs.update({
            "load_in_4bit": True,
            "quantization_config": {
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
        })
    
    # Load the student model
    logger.info(f"Loading student model: {config['student']['model_name']}...")
    student_model = AutoModelForCausalLM.from_pretrained(
        config["student"]["model_name"],
        **model_kwargs
    )
    
    # Apply LoRA if requested
    if args.use_lora:
        logger.info("Setting up LoRA for parameter-efficient fine-tuning...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()
    
    # Initialize the Gemini teacher model
    logger.info(f"Initializing Gemini teacher model: {config['gemini']['model_name']}...")
    teacher_config = GeminiConfig(
        api_key=api_key,
        model_name=config["gemini"]["model_name"],
        cache_dir=config["distillation"]["cache_dir"],
        max_api_retries=config["distillation"]["max_api_retries"],
        api_timeout=config["distillation"]["api_timeout"],
        vocab_size=len(tokenizer)  # Match vocabulary size of student
    )
    teacher_model = GeminiTeacherWrapper(
        config=teacher_config,
        student_tokenizer=tokenizer
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        save_total_limit=config["training"].get("save_total_limit", 3),
        group_by_length=config["training"].get("group_by_length", True),
        report_to="wandb",
        run_name=f"gemini-distillation-{config['student']['model_name'].split('/')[-1]}",
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        evaluation_strategy="steps",
        eval_steps=config["training"]["save_steps"],
        load_best_model_at_end=True,
    )
    
    # Initialize the Trainer
    trainer = GeminiDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        alpha=config["distillation"]["alpha"],
        temperature=config["distillation"]["temperature"],
    )
    
    # Train the model
    logger.info("Starting training...")
    resume_checkpoint = config["training"].get("resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save the final model
    logger.info(f"Saving the final model to {config['training']['output_dir']}...")
    trainer.save_model(config["training"]["output_dir"])
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
