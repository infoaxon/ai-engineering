# cdp_db_import.py

import sqlite3
import pandas as pd
import numpy as np
from faker import Faker

# Configuration
NUM_CUSTOMERS = 500
SEGMENTS = [
    "HighValueEngaged",
    "AtRiskLapsed",
    "YoungUrban",
    "FamilyBundlers",
    "DormantProspect",
    "SeniorStrained",
    "CampaignResponsive",
]
DB_FILE = "cdp_segments_demo.db"

# Initialize Faker
fake = Faker()

# Generate synthetic CustomerMaster data
customer_records = []
for i in range(NUM_CUSTOMERS):
    record_id = f"CID{i:04d}"
    customer_records.append(
        {
            "recordId": record_id,
            "name": fake.name(),
            "email": fake.email(),
            "age": np.random.randint(18, 80),
            "lastPolicyEndDate": fake.date_between(
                start_date="-2y", end_date="today"
            ).isoformat(),
        }
    )
customer_df = pd.DataFrame(customer_records)

# Compute daysSinceLastPolicy
customer_df["daysSinceLastPolicy"] = (
    pd.Timestamp.now() - pd.to_datetime(customer_df["lastPolicyEndDate"])
).dt.days

# Generate synthetic UserActionProfile
action_df = pd.DataFrame(
    {
        "recordId": customer_df["recordId"],
        "numOfLogins30d": np.random.poisson(5, size=NUM_CUSTOMERS),
    }
)

# Generate synthetic LifetimeMetrics
metrics_df = pd.DataFrame(
    {
        "recordId": customer_df["recordId"],
        "totalCustomerLifetimeValue": np.random.gamma(
            2, 50000, size=NUM_CUSTOMERS
        ).round(2),
        "propensityScore": np.clip(
            np.random.beta(2, 5, size=NUM_CUSTOMERS), 0, 1
        ).round(2),
    }
)

# Assign random segments
segment_choices = np.random.choice(
    SEGMENTS, size=NUM_CUSTOMERS, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10]
)
segments_df = pd.DataFrame(
    {
        "recordId": customer_df["recordId"],
        "segmentName": segment_choices,
        "daysSinceLastPolicy": customer_df["daysSinceLastPolicy"],
    }
)

# Write to SQLite
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Drop view if exists to allow table overwrite
cur.execute("DROP VIEW IF EXISTS CustomerMasterWithDays;")
conn.commit()

# Overwrite/Create tables
customer_df[["recordId", "name", "email", "age", "lastPolicyEndDate"]].to_sql(
    "CustomerMaster", conn, if_exists="replace", index=False
)
customer_df[["recordId", "daysSinceLastPolicy"]].to_sql(
    "CustomerMasterWithDays", conn, if_exists="replace", index=False
)
action_df.to_sql("UserActionProfile", conn, if_exists="replace", index=False)
metrics_df.to_sql("LifetimeMetrics", conn, if_exists="replace", index=False)
segments_df.to_sql("CustomerSegments", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print(f"Synthetic dataset loaded into '{DB_FILE}' successfully!")
