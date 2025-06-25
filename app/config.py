import os
from dotenv import load_dotenv

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ETHER_VALUE = 10 ** 18

BASE_URL = "https://api.etherscan.io/api"
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"