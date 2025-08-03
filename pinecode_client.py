import os
from pinecone import Pinecone, ServerlessSpec

# Use your actual API key here (or load from env var)
api_key = "pcsk_6x2Una_4FouFhZ3gWHFYWuqBnY5TsYCaGNyGc51NmWtPwBMGrgVHcEisXMXXWrgLM7f4Gx"

pc = Pinecone(api_key=api_key)

# Check if index exists
if 'my-index' not in pc.list_indexes().names():
    pc.create_index(
        name='my-index',
        dimension=1536,  # match your embedding size
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # match your region
        )
    )

# Connect to the index
index = pc.Index("my-index")

# Example upsert
index.upsert([
    ("id-1", [0.1] * 1536),  # Replace with your actual vector
])

# Example query
result = index.query(
    vector=[0.1] * 1536,  # Replace with query embedding
    top_k=3,
    include_metadata=True
)

print(result)
