from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI()
# ensure some of the sample data exists
df = pd.read_csv("cdp_master_synthetic.csv")


@app.get("/customers/to_nudge")
def to_nudge(segment: str):
    subset = df[df["segmentFlag"] == segment]
    if subset.empty:
        raise HTTPException(404, f"No customers in segment {segment}")
    return subset.to_dict(orient="records")


# Run with:
#   uvicorn data_api:app --reload
