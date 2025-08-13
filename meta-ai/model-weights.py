import yaml
import requests

# Getting the model architecture from the Ollama API from my local model here


def get_model_info(model="llama3"):
    res = requests.post("http://localhost:11434/api/show", json={"name": model})
    return res.json()


# And now I am printing what the API returns above, I will print it in YAML


info = get_model_info("llama3.2")
print(yaml.dump(info, sort_keys=False, default_flow_style=False))
