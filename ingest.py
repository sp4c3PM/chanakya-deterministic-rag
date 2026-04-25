try:
    import truststore; truststore.inject_into_ssl()  # corporate SSL proxy (optional)
except ImportError:
    pass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.feature_extraction.text import TfidfVectorizer
import uuid
import hashlib
import json
from pathlib import Path

MANIFEST_FILE = Path("corpus_manifest.json")

def compute_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_corpus():
    if not MANIFEST_FILE.exists():
        return True, []
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    tampered = []
    for path_str, expected_hash in manifest.items():
        p = Path(path_str)
        if p.exists() and compute_checksum(p) != expected_hash:
            tampered.append(path_str)
    return len(tampered) == 0, tampered

def update_manifest(paths):
    manifest = {str(p): compute_checksum(p) for p in paths}
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

COLLECTION = "defence_docs"
CHUNK_SIZE = 200

import re

def detect_header(line):
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if line.isupper() and len(line) > 4:
        return True
    if re.match(r'^(Chapter|Section|Part|Clause|Para|Appendix|Schedule)\s+[\dIVXA-Z]', line, re.I):
        return True
    if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
        return True
    return False

def chunk_text(text, source, size=CHUNK_SIZE):
    lines = text.splitlines()
    chunks = []
    current_header = ""
    words_buffer = []
    chunk_id = 0

    def emit_buffer(header, force=False):
        nonlocal chunk_id
        while len(words_buffer) >= size:
            chunk_words = words_buffer[:size]
            del words_buffer[:size]
            body = " ".join(chunk_words)
            labeled = f"[{header}] {body}" if header else body
            chunks.append({"text": labeled, "source": source, "chunk_id": chunk_id, "header": header})
            chunk_id += 1
        if force and words_buffer:
            body = " ".join(words_buffer)
            labeled = f"[{header}] {body}" if header else body
            chunks.append({"text": labeled, "source": source, "chunk_id": chunk_id, "header": header})
            chunk_id += 1
            words_buffer.clear()

    for line in lines:
        if detect_header(line):
            # Flush ALL buffered content (including partial) under old header before switching
            emit_buffer(current_header, force=True)
            current_header = line.strip()
        words_buffer.extend(line.split())
        emit_buffer(current_header)

    emit_buffer(current_header, force=True)
    return chunks

def extract_pdf(path):
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"  [WARN] Could not parse {path}: {e}")
        return ""

def load_docs():
    chunks = []
    txt_paths = list(Path(".").glob("*.txt")) + list(Path("docs").glob("*.txt"))
    pdf_paths = list(Path(".").glob("*.pdf")) + list(Path("docs").glob("*.pdf"))
    all_paths = []

    # Verify existing corpus integrity
    ok, tampered = verify_corpus()
    if not ok:
        print(f"  [WARN] TAMPER DETECTED in: {', '.join(tampered)}")

    for path in txt_paths:
        text = path.read_text()
        if text.strip():
            chunks.extend(chunk_text(text, str(path)))
            all_paths.append(path)
            print(f"  Loaded: {path} ({len(text.split())} words)")

    for path in pdf_paths:
        text = extract_pdf(path)
        if len(text.split()) > 50:
            chunks.extend(chunk_text(text, str(path)))
            all_paths.append(path)
            print(f"  Loaded PDF: {path} ({len(text.split())} words)")
        else:
            print(f"  [SKIP] {path} — not a valid PDF or empty")

    update_manifest(all_paths)
    return chunks

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts).toarray()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=matrix.shape[1], distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=matrix[i].tolist(), payload=chunks[i])
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return client, vectorizer

if __name__ == "__main__":
    print("Loading docs...")
    chunks = load_docs()
    print(f"\nTotal chunks: {len(chunks)}")
    client, vectorizer = build_index(chunks)
    print(f"Index built. Dim: {vectorizer.transform([chunks[0]['text']]).toarray().shape[1]}")
    print("Done.")
