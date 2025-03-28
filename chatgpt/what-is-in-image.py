from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.shutterstock.com/shutterstock/photos/2362192535/display_1500/stock-photo-a-proof-of-automobile-insurance-card-that-is-a-mock-generic-card-is-seen-in-a-d-illustration-2362192535.jpg"
                    },
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
