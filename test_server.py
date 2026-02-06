import torch
import torch.nn.functional as F
import struct
import requests
import json

# Load a 木 sample from test data
data = torch.load('data/processed/test_shard0.pt', weights_only=False)
labels = data['labels']
features = data['features']
matches = (labels == 1781).nonzero(as_tuple=True)[0]

# Get first 木 sample
sample = features[matches[0]].numpy().tolist()

print("Sending actual training data to server...")
print(f"Shape: {len(sample)} x {len(sample[0])}")

# Send to server
response = requests.post('http://localhost:5000/predict',
                         json={'features': sample},
                         headers={'Content-Type': 'application/json'})

result = response.json()
print("\nServer response:")
if result['status'] == 'success':
    for m in result['matches']:
        print(f"  {m['char']}: {m['dist']}")
else:
    print(f"Error: {result['message']}")
