import requests
import numpy as np

SERVER_URL = "http://localhost:8002"

def test_single():
    payload = {
        "prompt": "A photo of a dog running on the grass"
    }

    r = requests.post(f"{SERVER_URL}/embed", json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)

    print("=== Single Embedding Test ===")
    print("Dimension:", data["dim"])
    print("L2 norm:", np.linalg.norm(embedding))
    print("Max:", embedding.max())
    print("Min:", embedding.min())

    assert embedding.shape[0] == data["dim"]
    assert not np.allclose(embedding, 0)
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-3

    print("✅ Single embedding OK")


if __name__ == "__main__":
    test_single()
