import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

INDEX_NAME = "pmars-social-posts"
EMBED_MODEL = "text-embedding-3-small"
DIMENSION = 1536

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "cleaned_chunks.json")


def embed_texts(client, texts):
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_or_create_index(pc):
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Waiting for index to be ready", end="", flush=True)
        for _ in range(30):
            if pc.describe_index(INDEX_NAME).status["ready"]:
                break
            print(".", end="", flush=True)
            time.sleep(1)
        print(" ready.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    return pc.Index(INDEX_NAME)


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    texts = [c["cleaned_text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    embeddings = embed_texts(client, texts)

    index = get_or_create_index(pc)

    vectors = [
        (
            str(c["chunk_id"]),
            embeddings[i],
            {
                "chunk_id": c["chunk_id"],
                "pid": c["pid"],
                "title": c["title"],
                "tags": c["tags"],
                "cleaned_text": c["cleaned_text"],
            },
        )
        for i, c in enumerate(chunks)
    ]

    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} vectors into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
