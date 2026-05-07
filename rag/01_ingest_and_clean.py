import json
import os
import re
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "social_media_db"
COLLECTION_NAME = "posts"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "cleaned_chunks.json")
POSTS_PATH = os.path.join(os.path.dirname(__file__), "cleaned_posts.json")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Regex patterns for PII scrubbing (pattern-based redaction)
PII_PATTERNS = [
    (re.compile(r"@\w+"), ""),                                                    # @mentions
    (re.compile(r"https?://\S+"), ""),                                            # URLs
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"), ""),                     # emails
    (re.compile(r"\b(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), ""), # phone numbers
    (re.compile(r"\s{2,}"), " "),                                                 # collapse extra whitespace
]

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def scrub_pii(text):
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


def clean_post(doc):
    """Full cleaned post with no chunking — used as the source of truth for LLM context."""
    pid = doc["pid"]
    title = scrub_pii(doc.get("title", ""))
    text = scrub_pii(doc.get("text", ""))
    img_desc = scrub_pii(doc.get("img_desc", ""))
    tags = doc.get("tags", [])
    cleaned_text = f"{title}\n{text}\n{img_desc}".strip()
    return {"pid": pid, "title": title, "tags": tags, "cleaned_text": cleaned_text}


def chunk_document(doc):
    pid = doc["pid"]
    title = scrub_pii(doc.get("title", ""))
    text = scrub_pii(doc.get("text", ""))
    img_desc = scrub_pii(doc.get("img_desc", ""))
    tags = doc.get("tags", [])

    full_text = f"{text}\n{img_desc}".strip()
    sub_chunks = splitter.split_text(full_text)

    chunks = []
    for i, sub in enumerate(sub_chunks):
        chunk_id = pid if len(sub_chunks) == 1 else f"{pid}_{i}"
        chunks.append({
            "chunk_id": chunk_id,
            "pid": pid,
            "title": title,
            "tags": tags,
            "cleaned_text": f"{title}\n{sub}",
        })
    return chunks


def main():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client[DB_NAME]
    posts = list(db[COLLECTION_NAME].find({}))
    print(f"Fetched {len(posts)} documents from MongoDB.")

    all_chunks = []
    cleaned_posts = []
    for doc in posts:
        all_chunks.extend(chunk_document(doc))
        cleaned_posts.append(clean_post(doc))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False, default=str)

    with open(POSTS_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_posts, f, indent=2, ensure_ascii=False, default=str)

    print(f"Wrote {len(all_chunks)} chunks from {len(posts)} posts to {OUTPUT_PATH}")
    print(f"Wrote {len(cleaned_posts)} full posts to {POSTS_PATH}")


if __name__ == "__main__":
    main()
