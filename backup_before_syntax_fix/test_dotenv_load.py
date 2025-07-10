import os

from dotenv import load_dotenv

# Explicitly load the .env file from the current working directory
load_dotenv(override=True)  # override=True ensures it reloads even if already loaded

print(f"DEBUG: Current Working Directory: {os.getcwd()}")
print(f"DEBUG: Value of NEWS_API_KEY: {os.getenv('NEWS_API_KEY')}")

# Optional: List files in CWD to confirm .env presence
# print(f"DEBUG: Files in CWD: {os.listdir()}")
