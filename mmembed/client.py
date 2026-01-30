import requests
import numpy as np

SERVER_URL = "http://localhost:8002"

def test_single():
    payload = {
        "instruction": "Express this document.",
        "text": "A photo of a dog running on the grass",
        "img_path": "/dataset/crawl/mmqa_image/'67_Austin_Mini_Moke_(Ottawa_British_Car_Show_'10).jpg",
    }

    r = requests.post(f"{SERVER_URL}/embed", json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    embedding = np.array(data["embedding"], dtype=np.float32)

    print("Single Embedding Test")
    print("L2 norm:", np.linalg.norm(embedding))
    print("Max:", embedding.max())
    print("Min:", embedding.min())

    assert not np.allclose(embedding, 0)
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-3

    print("Single embedding OK")


if __name__ == "__main__":
    test_single()
