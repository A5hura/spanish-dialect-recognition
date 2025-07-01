import requests
import openai
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import json

# === STEP 0: API SETUP ===
subscription_key = "YOUR_AZURE_SUBSCRIPTION_KEY"
region = "YOUR_AZURE_REGION"  # e.g., "eastus"
endpoint = "https://api.cognitive.microsofttranslator.com"

language_map = {
    "ARG": "es-AR",
    "CHI": "es-CL",
    "COL": "es-CO",
    "MEX": "es-MX",
    "BRA": "pt-BR"
}

headers_azure = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Ocp-Apim-Subscription-Region': region,
    'Content-type': 'application/json'
}

params_azure = {
    'api-version': '3.0',
    'to': 'en'
}

# === OpenAI API Key ===
root = tk.Tk()
root.withdraw()
key_file_path = filedialog.askopenfilename(
    title="Select OpenAI API Key File",
    filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
)

if not key_file_path or not os.path.exists(key_file_path):
    print("❌ No OpenAI API key file selected. Exiting.")
    exit(1)

with open(key_file_path, "r", encoding="utf-8") as f:
    openai.api_key = f.read().strip()

client = openai.OpenAI(api_key=openai.api_key)

# === STEP 1: Read Data ===
input_file = "labeled_output.txt"
data = []

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        if "=>" not in line:
            continue
        sentence, label = line.rsplit("=>", 1)
        sentence = sentence.strip()
        label = label.strip().upper()
        data.append((sentence, label))

# === STEP 2 & 3: Translate using Azure and OpenAI ===
results = []

for sentence, label in data:
    source_lang = language_map.get(label, "es")

    # Azure Translation
    try:
        body = [{'text': sentence}]
        response = requests.post(
            f"{endpoint}/translate",
            params={**params_azure, 'from': source_lang},
            headers=headers_azure,
            json=body
        )
        translated_azure = response.json()[0]['translations'][0]['text']
    except Exception as e:
        print(f"Azure error: {e}")
        translated_azure = "ERROR"

    # OpenAI Translation
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Spanish/Portuguese-to-English translator."},
                {"role": "user", "content": f"Translate to English:\n{sentence}"}
            ],
            temperature=0,
            max_tokens=100
        )
        translated_openai = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        translated_openai = "ERROR"

    results.append({
        "Original": sentence,
        "Azure_English": translated_azure,
        "OpenAI_English": translated_openai,
        "Country": label
    })

# === STEP 4: Save to Excel ===
df = pd.DataFrame(results)
df.to_excel("translated_output.xlsx", index=False)

print("✅ Translations complete. Saved as 'translated_output.xlsx'")