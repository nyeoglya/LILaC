import requests

SERVER_URL = "http://localhost:8003"

def test_single():
    payload = {
        "instruction": "Instruction: Represent the text for retrieval.",
        "text": "Introduce yourself.",
        "image_path": "",
    }

    r = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    print(data["response"])


if __name__ == "__main__":
    test_single()
