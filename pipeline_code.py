import pandas as pd
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential

# --- CONFIGURATION ---
AZURE_KEY = "your_azure_key_here"
ENDPOINT = "your_endpoint_here"
REGION = "northcentralus"

# --- LANGUAGE MAPPING ---
language_map = {
    "BRA": "pt",        # Brazilian Portuguese
    "MEX": "es",        # Mexican Spanish
    "COL": "es",        # Colombian Spanish
    "CHL": "es",        # Chilean Spanish
    "ARG": "es",        # Argentinian Spanish
}

# --- SETUP AZURE TRANSLATOR ---
translator = TextTranslationClient(
    endpoint=ENDPOINT,
    region=REGION,
    credential=AzureKeyCredential(AZURE_KEY)
)

# --- READ CSV ---
df = pd.read_csv("your_input_file.csv")  # 2 columns: [Information, Country_Code]

# --- TRANSLATE EACH SENTENCE ---
def translate_text(sentence, country_code):
    if pd.isna(sentence) or pd.isna(country_code):
        return ""
    src_lang = language_map.get(country_code.strip().upper(), "auto")
    try:
        response = translator.translate(
            content=[sentence],
            from_parameter=src_lang,
            to=["en"]
        )
        for item in response:
            for t in item.translations:
                return t.text
    except Exception as e:
        print(f"Error translating '{sentence}': {e}")
        return "ERROR"

# Apply translation
df["Translated_English"] = df.apply(lambda row: translate_text(row[0], row[1]), axis=1)

# --- SAVE OUTPUT ---
df.to_csv("translated_output.csv", index=False)
