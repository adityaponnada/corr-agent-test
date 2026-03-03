import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# from dotenv import load_dotenv



import os
import google.generativeai as genai

# MANUALLY PASTE YOUR KEY HERE (Just for this test!)
MY_KEY = "AIzaSyAQCA8ID8UhmtBMfMVigUYqMO688lwqi7U" # Your actual API key from Google AI Studio

print("🔗 Attempting manual connection...")
genai.configure(api_key=MY_KEY)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Available: {m.name}")
except Exception as e:
    print(f"❌ Connection failed: {e}")


# # 1. This "wakes up" the .env file
# load_dotenv() 

# api_key = os.getenv("GOOGLE_API_KEY")

# if api_key:
#     print("🔍 Checking available models for your API key...")
#     try:
#         for m in genai.list_models():
#             if 'generateContent' in m.supported_generation_methods:
#                 print(f"✅ Available: {m.name}")
#     except Exception as e:
#         print(f"❌ Error listing models: {e}")

if MY_KEY:
    print("✅ API Key found!")
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=MY_KEY)
    # A quick test pull
    try:
        response = llm.invoke("Say 'Hello World'")
        print("🤖 Model Response:", response.content)
    except Exception as e:
        print("❌ Error connecting to Gemini:", e)
else:
    print("❌ API Key NOT found. Check your export command.")