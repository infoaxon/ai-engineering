import streamlit as st
import pandas as pd
import requests
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import threading

# --- Sample customer dataset ---
data = {
    "customer_id": [101, 102, 103, 104],
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [35, 45, 29, 52],
    "location": ["Mumbai", "Delhi", "Bangalore", "Chennai"],
    "last_login_days_ago": [40, 5, 70, 15],
    "active_policies": [["Health", "Motor"], ["Life"], ["Motor"], ["Health", "Home"]],
    "policy_expiring_in_days": [15, 60, 5, 90],
    "missed_notifications": [3, 0, 5, 1],
    "email_engagement_score": [0.2, 0.9, 0.1, 0.6],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# --- Rule-based recommendation engine ---
def get_recommendations(customer):
    recs = []

    if customer['policy_expiring_in_days'] <= 30:
        recs.append({
            "recommendation": "🔄 Renew your policy before it expires to avoid coverage gaps.",
            "why": f"The policy is expiring in {customer['policy_expiring_in_days']} days."
        })

    if customer['last_login_days_ago'] > 30 or customer['missed_notifications'] > 2:
        recs.append({
            "recommendation": "📬 You've missed some important updates. Catch up now!",
            "why": f"Last login was {customer['last_login_days_ago']} days ago and missed {customer['missed_notifications']} notifications."
        })

    if customer['email_engagement_score'] < 0.3:
        recs.append({
            "recommendation": "💡 Don't miss out! Check your email for exclusive offers.",
            "why": f"Engagement score is low at {customer['email_engagement_score']}."
        })

    policies = customer['active_policies']
    if "Motor" in policies and "Health" not in policies:
        recs.append({
            "recommendation": "❤️ Secure your health with our affordable health plans.",
            "why": "Customer has Motor insurance but no Health plan."
        })
    if "Health" in policies and "Life" not in policies:
        recs.append({
            "recommendation": "🌟 Protect your family's future with a life insurance plan.",
            "why": "Customer has Health plan but no Life insurance."
        })

    return recs

# --- LLM Integration with Ollama ---
def query_llama(customer_profile):
    prompt = f"""
    You are an insurance assistant. Based on the following customer profile, suggest the most relevant next best actions or offers:

    {json.dumps(customer_profile, indent=2)}

    Provide 2-3 actionable, customer-friendly suggestions.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json().get("response", "No response from LLM")
        else:
            return f"Error from Ollama: {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# --- FastAPI Setup ---
app = FastAPI()

class CustomerProfile(BaseModel):
    customer_id: int
    name: str
    age: int
    location: str
    last_login_days_ago: int
    active_policies: list
    policy_expiring_in_days: int
    missed_notifications: int
    email_engagement_score: float

@app.post("/recommendations/rules")
async def rules_recommendation(profile: CustomerProfile):
    return {"recommendations": get_recommendations(profile.model_dump())}

@app.post("/recommendations/llm")
async def llm_recommendation(profile: CustomerProfile):
    return {"llm_suggestions": query_llama(profile.model_dump())}

# Run FastAPI in background
threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error"), daemon=True).start()

# --- Streamlit App ---
st.title("🧠 Insurance Next Best Action Demo")
st.write("Select a customer to see personalized recommendations:")

customer_name = st.selectbox("Choose a customer:", df['name'])
selected_customer = df[df['name'] == customer_name].iloc[0]

st.subheader("📄 Customer Profile")
st.json(selected_customer.to_dict())

st.subheader("✅ Rule-based Recommended Actions")
recommendations = get_recommendations(selected_customer)
if recommendations:
    for rec in recommendations:
        st.markdown(f"- **{rec['recommendation']}**\n  ⚡ _Why_: {rec['why']}")
else:
    st.info("No recommendations at this time.")

st.subheader("🤖 LLM-based Suggestions")
if st.button("Ask LLaMA 3.2 (Ollama)"):
    with st.spinner("Thinking..."):
        llama_response = query_llama(selected_customer.to_dict())
        st.markdown(f"**LLM Suggestion:**\n\n{llama_response}")

