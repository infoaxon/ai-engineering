# agent_ollama.py

import requests
import argparse
from datetime import datetime
from dateutil import parser as dtparser

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# 1) Define the prompt template
prompt = PromptTemplate(
    input_variables=[
        "name",
        "age",
        "daysSinceLastPolicy",
        "numOfLogins30d",
        "propensityScore",
    ],
    template="""\
You are writing a one-sentence reactivation nudge for an insurance customer.
- Name: {name}
- Age: {age}
- Days since last policy ended: {daysSinceLastPolicy}
- Logins in last 30 days: {numOfLogins30d}
- Propensity score (0-1): {propensityScore}

Craft a friendly, personalized message that includes at least two of those dynamic values.
Your nudge:""",
)

# 2) Swap in the Ollama LLM wrapper
llm = OllamaLLM(
    model="llama3.2:latest",  # exactly the name from `ollama ls`
    base_url="http://127.0.0.1:11434",
    temperature=0.7,
)


def fetch_customers(segment: str):
    resp = requests.get(f"http://localhost:8000/customers/to_nudge?segment={segment}")
    resp.raise_for_status()
    return resp.json()


def send_nudge(to: str, body: str):
    print(f"\n--- NUDGE ---\nTo: {to}\n{body}\n")


def generate_nudge(cust: dict) -> str:
    # compute days since last policy ended
    ended = dtparser.parse(cust["lastPolicyEndDate"])
    days_since = (datetime.now() - ended).days

    # format the prompt with dynamic fields
    text = prompt.format(
        name=cust["name"],
        age=cust["age"],
        daysSinceLastPolicy=days_since,
        numOfLogins30d=cust["numOfLogins30d"],
        propensityScore=cust["propensityScore"],
    )

    # call your local Llama 3.2 via Ollama
    result = llm.generate([text])
    return result.generations[0][0].text.strip()


def main(segment: str):
    customers = fetch_customers(segment)
    for cust in customers:
        message = generate_nudge(cust)
        send_nudge(cust["email"], message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required=True, help="Segment to nudge")
    args = parser.parse_args()
    main(args.segment)
