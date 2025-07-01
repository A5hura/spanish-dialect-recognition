import pandas as pd

# Set your file path (assumes file is in same directory as script)
input_file = "input.txt"
output_file = "output.xlsx"

# Read and parse lines
data = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        if '=>' in line:
            info, code = line.strip().split('=>')
            data.append((code.strip(), info.strip()))

# Create DataFrame
df = pd.DataFrame(data, columns=["Country_Code", "Information"])

# Save to Excel
df.to_excel(output_file, index=False)

print(f"✅ Excel file saved as: {output_file}")
