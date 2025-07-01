from pyspark.sql.functions import split, trim
import pandas as pd

# ✅ Read .txt file using DataFrame API
df_raw = spark.read.text("dbfs:/FileStore/my_data/input.txt")

# ✅ Parse 'INFORMATION => CODE' format
df_parsed = df_raw.withColumn("Information", trim(split(df_raw["value"], "=>")[0])) \
                  .withColumn("Country_Code", trim(split(df_raw["value"], "=>")[1])) \
                  .select("Country_Code", "Information")

# ✅ Convert to Pandas and save as Excel
pandas_df = df_parsed.toPandas()
pandas_df.to_excel("/dbfs/FileStore/my_data/output.xlsx", index=False)

print("✅ Excel saved at /dbfs/FileStore/my_data/output.xlsx")
