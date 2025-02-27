"""
Quick script to test the Gemini API connection
"""
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY is not set in the .env file")
    exit(1)

# Configure the API
genai.configure(api_key=api_key)

# Select the model
model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-thinking-exp-1219")
print(f"Testing Gemini API with model: {model_name}")

try:
    # Initialize the model
    model = genai.GenerativeModel(model_name)
    
    # Simple prompt
    prompt = "Hello, please provide a very short response to test the API connection."
    
    # Make the request
    print("Sending request to Gemini API...")
    start_time = time.time()
    response = model.generate_content(prompt)
    end_time = time.time()
    
    # Print the response
    print(f"\nResponse received in {end_time - start_time:.2f} seconds:")
    print(f"{response.text[:200]}...")
    
    print("\nAPI test successful! ✅")
    
except ResourceExhausted as e:
    print(f"\nAPI Rate limit exceeded: {e}")
    print("This is normal - the API has strict rate limits.")
    
except ServiceUnavailable as e:
    print(f"\nService unavailable: {e}")
    print("The API service may be temporarily unavailable.")
    
except Exception as e:
    print(f"\nError connecting to Gemini API: {e}")
    print("Please check your API key and internet connection.")
