import json
from doublewhammy import SynthDataGen

# Configuration
CONFIG_PATH = "config.json"
NUM_RECORDS = 100
OUTPUT_DIR = "./synthetic_output"

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    schema = load_config(CONFIG_PATH)
    generator = SynthDataGen()

    dataframes = {}
    for step in schema.get("steps", []):
        table_name = step["table_name"]
        columns = step["columns"]
        df = generator.generate_table(columns, NUM_RECORDS)
        dataframes[table_name] = df
        print(f"[✅ GENERATED] {table_name} ({len(df)} rows)")

    generator.export_to_csv(dataframes, OUTPUT_DIR)
    print(f"[💾 EXPORT COMPLETE] All tables saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
