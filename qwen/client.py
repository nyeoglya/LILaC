import requests

SERVER_URL = "http://localhost:8000"

def test_single():
    payload = {
        "instruction": "Instruction: Represent the text for retrieval.",
        "text": "Introduce yourself.",
        "img_paths": [],
    }

    r = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    print(data["response"])


if __name__ == "__main__":
    test_single()
