import requests
import argparse
import json

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RunnableSequence

# Your prompt template (unchanged)
prompt = PromptTemplate(
    input_variables=["name", "propensityScore"],
    template=(
        "Write a friendly, one-sentence nudge to re-activate an insurance policy.\n"
        "Customer name: {name}\n"
        "Propensity to buy (0-1): {propensityScore}\n"
        "Your message:"
    ),
)

# Initialize LLM and compose with the prompt
llm = OpenAI(temperature=0.7)
runnable = RunnableSequence(prompt, llm)


def fetch_customers(segment: str):
    resp = requests.get(f"http://localhost:8000/customers/to_nudge?segment={segment}")
    resp.raise_for_status()
    return resp.json()


def send_nudge(to: str, body: str):
    print(f"\n--- NUDGE ---\nTo: {to}\n{body}\n")


def main(segment: str):
    customers = fetch_customers(segment)
    for cust in customers:
        # Invoke the composed prompt+LLM
        out = runnable.invoke(
            {
                "name": cust["name"],
                "propensityScore": cust["propensityScore"],
            }
        )
        message = out.value.strip()
        send_nudge(cust["email"], message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required=True)
    args = parser.parse_args()
    main(args.segment)
