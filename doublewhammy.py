import os
import random
import string
import pandas as pd
import datetime
import calendar
from transformers import AutoModelForMaskedLM, AutoTokenizer
from nameparser import HumanName
from faker import Faker
import torch

class SynthDataGen:
    def __init__(self):
        self.fake = Faker()
        self.model_name = "prajjwal1/bert-tiny"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.currency_chart = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY", "SEK", "NZD"]

    def _generate_from_model(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        return self.tokenizer.decode(predicted_token_id).strip()

    def _generate_custom_id(self):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    def _generate_date(self, semantic_type):
        year = random.randint(1940, 2010)
        month = random.randint(1, 12)
        day = random.randint(1, calendar.monthrange(year, month)[1])
        return datetime.date(year, month, day)

    def _generate_date_dimension(self):
        start_date = datetime.date(2000, 1, 1)
        end_date = datetime.date(2030, 12, 31)
        date_range = pd.date_range(start_date, end_date)
        records = []
        for date in date_range:
            record = {
                "date_key": date.strftime("%Y%m%d"),
                "date": date.date(),
                "day_of_month": date.day,
                "day_of_week": date.weekday() + 1,
                "day_of_year": date.dayofyear,
                "week_of_month": (date.day - 1) // 7 + 1,
                "week_of_year": date.isocalendar()[1],
                "month_name": calendar.month_name[date.month],
                "month_number": date.month,
                "quarter": (date.month - 1) // 3 + 1,
                "year": date.year,
                "year_month": date.strftime("%Y-%m"),
                "is_weekend": date.weekday() >= 5,
            }
            records.append(record)
        return pd.DataFrame(records)

    def _generate_rating(self):
        return f"{random.randint(0, 100)}%"

    def _generate_phone(self):
        cc = f"+{random.randint(1, 99)}"
        num = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000,9999)}"
        return f"{cc}-{num}"

    def _generate_name(self, field):
        name = HumanName(self.fake.name())
        if field == "first_name":
            return name.first
        elif field == "last_name":
            return name.last
        elif field == "middle_name":
            return name.middle or "Lee"
        return name.full_name

    def _generate_project_name(self):
        adjectives = [
            "Red", "Blue", "Swift", "Silent", "Global", "Quantum", "Dynamic", "Smart", "Unified", "Bright"
        ]
        nouns = [
            "Falcon", "Horizon", "Stream", "Engine", "Bridge", "Pulse", "Vista", "Orbit", "Matrix", "Beacon"
        ]
        codename_style = f"Project {random.choice(adjectives)} {random.choice(nouns)}"

        functional_style = random.choice([
            f"{random.choice(['Customer', 'Finance', 'Ops', 'Sales'])}{random.choice(['Sync', '360', 'Portal'])}",
            f"{random.choice(['Data', 'Report', 'Cloud'])}{random.randint(1, 99)}",
            f"{random.choice(['Neo', 'Core', 'Edge'])} {random.randint(1, 5)}.{random.randint(0, 9)}"
        ])

        acronym_style = random.choice([
            "GRACE", "VISTA", "LUMEN", "CORE", "NOVA", "ZENITH"
        ])

        style = random.choice(["codename", "functional", "acronym"])

        if style == "codename":
            return codename_style
        elif style == "functional":
            return functional_style
        else:
            return f"Project {acronym_style}"

    def _generate_field(self, semantic_type, field):
        if semantic_type == "id":
            return self._generate_custom_id()
        if semantic_type == "first_name":
            return self._generate_name("first_name")
        if semantic_type == "last_name":
            return self._generate_name("last_name")
        if semantic_type == "middle_name":
            return self._generate_name("middle_name")
        if semantic_type == "birth_date" or semantic_type == "date":
            return self._generate_date(semantic_type)
        if semantic_type == "rating":
            return self._generate_rating()
        if semantic_type == "phone":
            return self._generate_phone()
        if semantic_type == "email":
            return self.fake.email()
        if semantic_type == "address":
            return self.fake.address().replace("\n", ", ")
        if semantic_type == "salary":
            return random.randint(30000, 150000)
        if semantic_type in ("amount", "price"):
            return round(random.uniform(10, 1000), 2)
        if semantic_type == "quantity":
            return random.randint(1, 20)
        if semantic_type == "currency":
            return random.choice(self.currency_chart)
        if semantic_type == "department":
            return random.choice(["Clothing", "Electronics", "Home", "Books", "Sports"])
        if semantic_type == "status":
            return random.choice(["Active", "Inactive", "Pending", "Completed"])
        if semantic_type == "product":
            return random.choice(["T-shirt", "Laptop", "Blender", "Book", "Football"])
        if semantic_type == "project":
            return self._generate_project_name()
        if semantic_type == "person":
            return self._generate_name("full_name")
        return self._generate_from_model(f"Generate a realistic {semantic_type} [MASK]")

    def generate_table(self, columns, num_rows):
        if "date_key" in columns:
            return self._generate_date_dimension()
        data = {col: [] for col in columns}
        for _ in range(num_rows):
            for col_name, meta in columns.items():
                semantic_type = meta.get("semantic_type", "")
                data[col_name].append(self._generate_field(semantic_type, col_name))
        return pd.DataFrame(data)

    def export_to_csv(self, tables: dict, path_prefix: str):
        os.makedirs(path_prefix, exist_ok=True)
        for table_name, df in tables.items():
            file_path = os.path.join(path_prefix, f"{table_name}.csv")
            df.to_csv(file_path, index=False)