from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Give a full summary of Indian Insurance Market in 2025"
        }
    ]
)

print(completion.choices[0].message.content)
