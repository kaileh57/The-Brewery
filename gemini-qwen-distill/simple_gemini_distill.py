"""
Simplified Gemini distillation script that's more memory efficient
and robust against crashes
"""
import os
import time
import json
import random
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

# Load environment variables
load_dotenv()

# Configuration from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env file")
    sys.exit(1)

GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-thinking-exp-1219")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "15"))  # Process fewer reasoning-focused samples
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./gemini_cache"))
MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 6  # seconds, respects rate limits

# Create cache directory
CACHE_DIR.mkdir(exist_ok=True)
responses_file = CACHE_DIR / "conversations.json"

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

def get_gemini_response(prompt, retries=MAX_RETRIES):
    """Get a response from Gemini with rate limiting and retries, including chain-of-thought reasoning"""
    # Add chain-of-thought instruction to encourage step-by-step reasoning
    cot_prompt = f"""
    Think through this step by step:
    
    {prompt}
    
    First, reason through this problem carefully, showing all your work and intermediate steps.
    Break down your thinking process explicitly.
    Then provide your final answer or conclusion.
    """
    
    # Ensure we respect rate limits (10 calls/minute)
    time.sleep(DELAY_BETWEEN_CALLS)
    
    for attempt in range(retries):
        try:
            response = gemini_model.generate_content(
                cot_prompt,
                generation_config={"temperature": 1.0}
            )
            return response.text
            
        except (ResourceExhausted, ServiceUnavailable) as e:
            # Rate limit error or service unavailable
            wait_time = DELAY_BETWEEN_CALLS * (2 ** attempt) + random.uniform(0, 1)
            print(f"API error: {e}. Retrying in {wait_time:.1f} seconds... ({attempt+1}/{retries})")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            wait_time = DELAY_BETWEEN_CALLS * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
    
    # If we've exhausted all retries
    return "I apologize, but I couldn't process that request due to technical limitations."

def main():
    """Main function to process examples and generate responses"""
    # Load existing conversations if available
    conversations = []
    if responses_file.exists():
        try:
            with open(responses_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            print(f"Loaded {len(conversations)} existing conversations from {responses_file}")
        except json.JSONDecodeError:
            print(f"Error loading {responses_file}, starting fresh")
            conversations = []
    
    # How many more conversations do we need?
    remaining = NUM_SAMPLES - len(conversations)
    
    if remaining <= 0:
        print(f"Already have {len(conversations)} conversations, no more needed")
        return
    
    print(f"Generating {remaining} new conversations...")
    for i in range(remaining):
        # Generate a reasoning-oriented question
        topics = [
            "Solve this probability problem: Three friends play a game where they each flip a coin. What's the probability that at least two of them get the same result?",
            "Evaluate this argument: 'All birds can fly. Penguins are birds. Therefore, penguins can fly.' Is this logically valid? Why or why not?",
            "How would you determine if a number is a prime number? Design an algorithm and trace through it for the number 29.",
            "Analyze the impact of increasing interest rates on inflation and unemployment. What are the trade-offs?",
            "A ball is thrown upward with an initial velocity of 20 m/s. How high will it go, and when will it hit the ground? Use g = 9.8 m/s².",
            "Explain the trolley problem in ethics and analyze the different philosophical perspectives on the right action.",
            "Design a system to efficiently store and retrieve large amounts of data. What are the key considerations and tradeoffs?",
            "How would you estimate the number of piano tuners in New York City? Break down your reasoning.",
            "Analyze how machine learning models can be biased. What are three sources of bias and how might they be mitigated?",
            "Compare and contrast different approaches to solving climate change. Which approach would be most effective and why?",
            "In a room of 30 people, what's the probability that at least two people share a birthday? Show your calculation.",
            "If you have 8 identical balls and 3 distinct boxes, in how many ways can you distribute the balls among the boxes?",
            "Explain how you would determine if a chemical reaction is endothermic or exothermic. What measurements would you take?",
            "Evaluate the most efficient sorting algorithm for different dataset sizes and characteristics.",
            "How would you design an experiment to test whether a plant grows better with classical music or rock music?"
        ]
        question = random.choice(topics)
        
        # Get response from Gemini
        print(f"Processing question {i+1}/{remaining}: {question[:50]}...")
        response = get_gemini_response(question)
        
        # Create conversation pair
        conversation = {
            "question": question,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to our collection
        conversations.append(conversation)
        
        # Save progress after each conversation
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        print(f"Progress: {len(conversations)}/{NUM_SAMPLES} conversations completed")
    
    print(f"Completed! Generated {len(conversations)} conversations.")
    print(f"Saved to: {responses_file}")
    print("Next steps:")
    print("1. Run the 'train_qwen.py' script to train the model on these conversations")

if __name__ == "__main__":
    main()
