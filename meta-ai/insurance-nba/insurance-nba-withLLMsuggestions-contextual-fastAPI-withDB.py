import streamlit as st
import pandas as pd
import requests
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import threading
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- SQLAlchemy setup for SQLite DB ---
Base = declarative_base()


class Customer(Base):
    __tablename__ = "customers"
    customer_id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)
    location = Column(String)
    last_login_days_ago = Column(Integer)
    active_policies = Column(JSON)
    policy_expiring_in_days = Column(Integer)
    missed_notifications = Column(Integer)
    email_engagement_score = Column(Float)


engine = create_engine("sqlite:///customers.db")
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# --- Seed data if DB is empty ---
session = SessionLocal()
if session.query(Customer).count() == 0:
    seed_data = [
        Customer(
            customer_id=101,
            name="Alice",
            age=35,
            location="Mumbai",
            last_login_days_ago=40,
            active_policies=["Health", "Motor"],
            policy_expiring_in_days=15,
            missed_notifications=3,
            email_engagement_score=0.2,
        ),
        Customer(
            customer_id=102,
            name="Bob",
            age=45,
            location="Delhi",
            last_login_days_ago=5,
            active_policies=["Life"],
            policy_expiring_in_days=60,
            missed_notifications=0,
            email_engagement_score=0.9,
        ),
        Customer(
            customer_id=103,
            name="Charlie",
            age=29,
            location="Bangalore",
            last_login_days_ago=70,
            active_policies=["Motor"],
            policy_expiring_in_days=5,
            missed_notifications=5,
            email_engagement_score=0.1,
        ),
        Customer(
            customer_id=104,
            name="Diana",
            age=52,
            location="Chennai",
            last_login_days_ago=15,
            active_policies=["Health", "Home"],
            policy_expiring_in_days=90,
            missed_notifications=1,
            email_engagement_score=0.6,
        ),
    ]
    session.add_all(seed_data)
    session.commit()
session.close()

# --- Helper functions ---


def get_customer_by_id(customer_id: int):
    session = SessionLocal()
    customer = (
        session.query(Customer).filter(Customer.customer_id == customer_id).first()
    )
    session.close()
    return customer


# --- Recommendation Engine ---


def get_recommendations(customer):
    recs = []

    if customer.policy_expiring_in_days <= 30:
        recs.append(
            {
                "recommendation": "🔄 Renew your policy before it expires to avoid coverage gaps.",
                "why": f"The policy is expiring in {customer.policy_expiring_in_days} days.",
            }
        )

    if customer.last_login_days_ago > 30 or customer.missed_notifications > 2:
        recs.append(
            {
                "recommendation": "📬 You've missed some important updates. Catch up now!",
                "why": f"Last login was {customer.last_login_days_ago} days ago and missed {customer.missed_notifications} notifications.",
            }
        )

    if customer.email_engagement_score < 0.3:
        recs.append(
            {
                "recommendation": "💡 Don't miss out! Check your email for exclusive offers.",
                "why": f"Engagement score is low at {customer.email_engagement_score}.",
            }
        )

    policies = customer.active_policies
    if "Motor" in policies and "Health" not in policies:
        recs.append(
            {
                "recommendation": "❤️ Secure your health with our affordable health plans.",
                "why": "Customer has Motor insurance but no Health plan.",
            }
        )
    if "Health" in policies and "Life" not in policies:
        recs.append(
            {
                "recommendation": "🌟 Protect your family's future with a life insurance plan.",
                "why": "Customer has Health plan but no Life insurance.",
            }
        )

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
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
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


@app.get("/customer/{customer_id}")
async def get_customer(customer_id: int):
    customer = get_customer_by_id(customer_id)
    if customer:
        return customer.__dict__
    return {"error": "Customer not found"}


# Run FastAPI in background
threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error"),
    daemon=True,
).start()

# --- Streamlit App ---
st.title("🧠 Insurance Next Best Action Demo")
st.write("Select a customer to see personalized recommendations:")

session = SessionLocal()
all_customers = session.query(Customer).all()
session.close()
customer_names = [c.name for c in all_customers]
customer_name = st.selectbox("Choose a customer:", customer_names)
selected_customer = next(c for c in all_customers if c.name == customer_name)

st.subheader("📄 Customer Profile")
st.json(
    {
        "customer_id": selected_customer.customer_id,
        "name": selected_customer.name,
        "age": selected_customer.age,
        "location": selected_customer.location,
        "last_login_days_ago": selected_customer.last_login_days_ago,
        "active_policies": selected_customer.active_policies,
        "policy_expiring_in_days": selected_customer.policy_expiring_in_days,
        "missed_notifications": selected_customer.missed_notifications,
        "email_engagement_score": selected_customer.email_engagement_score,
    }
)

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
        llama_response = query_llama(
            {
                "customer_id": selected_customer.customer_id,
                "name": selected_customer.name,
                "age": selected_customer.age,
                "location": selected_customer.location,
                "last_login_days_ago": selected_customer.last_login_days_ago,
                "active_policies": selected_customer.active_policies,
                "policy_expiring_in_days": selected_customer.policy_expiring_in_days,
                "missed_notifications": selected_customer.missed_notifications,
                "email_engagement_score": selected_customer.email_engagement_score,
            }
        )
        st.markdown(f"**LLM Suggestion:**\n\n{llama_response}")
