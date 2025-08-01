import streamlit as st
import pandas as pd
import requests
import json

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

    # Renewal Reminder
    if customer['policy_expiring_in_days'] <= 30:
        recs.append(
            " Renew your policy before it expires to avoid coverage gaps.")

    # Re-engagement
    if customer['last_login_days_ago'] > 30 or customer['missed_notifications'] > 2:
        recs.append(" You've missed some important updates. Catch up now!")

    # Low engagement follow-up
    if customer['email_engagement_score'] < 0.3:
        recs.append(" Don't miss out! Check your email for exclusive offers.")

    # Cross-sell opportunities
    policies = customer['active_policies']
    if "Motor" in policies and "Health" not in policies:
        recs.append(" Secure your health with our affordable health plans.")
    if "Health" in policies and "Life" not in policies:
        recs.append(" Protect your family's future with a life insurance plan.")

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


# --- Streamlit App ---
st.title("Insurance Next Best Action Demo")
st.write("Select a customer to see personalized recommendations:")

# Select customer by name
customer_name = st.selectbox("Choose a customer:", df['name'])
selected_customer = df[df['name'] == customer_name].iloc[0]

# Show customer details
st.subheader(" Customer Profile")
st.json(selected_customer.to_dict())

# Show recommendations
st.subheader(" Rule-based Recommended Actions")
recommendations = get_recommendations(selected_customer)

if recommendations:
    for rec in recommendations:
        st.markdown(f"- {rec}")
else:
    st.info("No recommendations at this time.")

# Ask LLM for suggestions
st.subheader("🤖 LLM-based Suggestions")
if st.button("Ask LLaMA 3.2 (Ollama)"):
    with st.spinner("Thinking..."):
        llama_response = query_llama(selected_customer.to_dict())
        st.markdown(f"**LLM Suggestion:**\n\n{llama_response}")
