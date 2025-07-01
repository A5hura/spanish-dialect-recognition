import openai
import requests
import pandas as pd

# === Step 1: Insert Your API Keys ===
AZURE_KEY = "YOUR_AZURE_KEY_HERE"
AZURE_REGION = "YOUR_REGION"  # e.g., "eastus"
OPENAI_KEY = "YOUR_OPENAI_KEY_HERE"

# Set OpenAI API key
openai.api_key = OPENAI_KEY

# === Step 2: Dialect Code to Language Mapping ===
dialect_lang_map = {
    "ARG": "es-ar",
    "CHL": "es-cl",
    "COL": "es-co",
    "MEX": "es-mx",
    "BRA": "pt-br"
}

# === Step 3: Load the labeled input data ===
input_file = "labeled_output.txt"
rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if "=>" in line:
            sentence, code = line.strip().split("=>")
            sentence = sentence.strip()
            code = code.strip().upper()
            rows.append((sentence, code))

# === Step 4: Azure Translate Function ===
def azure_translate(text, source_lang):
    endpoint = "https://api.cognitive.microsofttranslator.com/translate"
    params = {
        "api-version": "3.0",
        "from": source_lang,
        "to": "en"
    }
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_REGION,
        "Content-Type": "application/json"
    }
    body = [{"text": text}]
    try:
        response = requests.post(endpoint, params=params, headers=headers, json=body)
        response.raise_for_status()
        return response.json()[0]["translations"][0]["text"]
    except Exception as e:
        print(f"Azure Error for '{text}': {e}")
        return "ERROR"

# === Step 5: OpenAI Translate Function ===
def openai_translate(text, source_lang):
    prompt = f"Translate the following sentence from {source_lang} to English:\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Error for '{text}': {e}")
        return "ERROR"

# === Step 6: Translation Pipeline ===
output_data = []
for sentence, code in rows:
    lang_code = dialect_lang_map.get(code, "es")
    azure_result = azure_translate(sentence, lang_code)
    openai_result = openai_translate(sentence, lang_code)
    output_data.append([sentence, code, azure_result, openai_result])

# === Step 7: Save to Excel ===
df = pd.DataFrame(output_data, columns=["Original", "Dialect", "Azure Translation", "OpenAI Translation"])
df.to_excel("translated_output.xlsx", index=False)
print("✅ Translations completed. Output saved to 'translated_output.xlsx'")
