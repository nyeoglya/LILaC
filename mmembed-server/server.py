import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import requests

# Each query needs to be accompanied by an corresponding instruction describing the task. 
task_name_to_instruct = {"example": "Retrieve a Wikipedia paragraph that provides an answer to the given query about the image."}

img1_url = 'https://cdn.contexttravel.com/image/upload/w_1500,q_60/v1574869648/blog/Facts%20about%20the%20Eiffel%20Tower/eiffelhero.jpg'
img2_url = 'https://trumpwhitehouse.archives.gov/wp-content/uploads/2021/01/40508989563_514189250a_o-1500x720.jpg'

instruction = task_name_to_instruct['example']
queries = [
    {'txt': 'What country does this place belong to?', 'img': Image.open(requests.get(img1_url, stream=True).raw)}, 
    {'txt': 'What country does this place belong to?', 'img': Image.open(requests.get(img2_url, stream=True).raw)},
]

# No instruction needed for retrieval passages
passages = [
    {'txt': "France, officially the French Republic, is a country located primarily in Western Europe. Its overseas regions and territories include French Guiana in South America, Saint Pierre and Miquelon in the North Atlantic, the French West Indies, and many islands in Oceania and the Indian Ocean, giving it one of the largest discontiguous exclusive economic zones in the world. Metropolitan France shares borders with Belgium and Luxembourg to the north, Germany to the northeast, Switzerland to the east, Italy and Monaco to the southeast, Andorra and Spain to the south, and a maritime border with the United Kingdom to the northwest. Its metropolitan area extends from the Rhine to the Atlantic Ocean and from the Mediterranean Sea to the English Channel and the North Sea. Its eighteen integral regions (five of which are overseas) span a combined area of 643,801 km2 (248,573 sq mi) and have a total population of 68.4 million as of January 2024. France is a semi-presidential republic with its capital in Paris, the country's largest city and main cultural and commercial centre."},
    {'txt': "The United States of America (USA), commonly known as the United States (U.S.) or America, is a country primarily located in North America. It is a federal union of 50 states and a federal capital district, Washington, D.C. The 48 contiguous states border Canada to the north and Mexico to the south, with the states of Alaska to the northwest and the archipelagic Hawaii in the Pacific Ocean. The United States also asserts sovereignty over five major island territories and various uninhabited islands. The country has the world's third-largest land area, largest exclusive economic zone, and third-largest population, exceeding 334 million. Its three largest metropolitan areas are New York, Los Angeles, and Chicago, and its three most populous states are California, Texas, and Florida."},
]

# load model with tokenizer
model = AutoModel.from_pretrained('nvidia/MM-Embed', trust_remote_code=True)
model = model.cuda()

# get the embeddings, the output embeddings are normalized to one
max_length = 4096
query_embeddings = model.encode(queries, is_query=True, instruction=instruction, max_length=max_length)['hidden_states']
passage_embeddings = model.encode(passages, max_length=max_length)['hidden_states']

# compute relevance scores
scores = (query_embeddings @ passage_embeddings.T) * 100
print(scores.tolist())
#[[31.019872665405273, 12.753520965576172], [11.135049819946289, 22.12639617919922]]
