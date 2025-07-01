from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col
import pandas as pd

# Start Spark session (already available in Databricks)
spark = SparkSession.builder.getOrCreate()

# Path to your text file in DBFS
file_path = "/dbfs/FileStore/my_data/input.txt"  # Update this as needed

# Load as RDD and convert to DataFrame
rdd = spark.sparkContext.textFile(file_path)

# Convert => to columns using DataFrame transformations
df = rdd.filter(lambda line: "=>" in line) \
        .map(lambda line: line.split("=>")) \
        .map(lambda parts: (parts[1].strip(), parts[0].strip())) \
        .toDF(["Country_Code", "Information"])

# Optional: Convert to Pandas and save to Excel
pandas_df = df.toPandas()
pandas_df.to_excel("/dbfs/FileStore/my_data/output.xlsx", index=False)

print("Excel file written to: /dbfs/FileStore/my_data/output.xlsx")
