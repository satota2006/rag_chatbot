import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

pdf_path = "sample.pdf"

# PDFからテキスト抽出
with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

# チャンク分割
def split_into_chunks(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = split_into_chunks(all_text)

# 埋め込みモデルを読み込み、ベクトル化
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(chunks)

# FAISSインデックスを作成し、ベクトルを追加
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

print(f"FAISSに保存されたベクトル数: {index.ntotal}")

# ここから検索テスト
query = "予習の方法を教えて"  # ← ここを自分の資料の内容に合わせて自由に変えてOK
query_vector = model.encode([query])

# 一番似ているチャンクを3件検索
k = 3
distances, indices = index.search(np.array(query_vector).astype('float32'), k)

print(f"\n質問: {query}")
print("---検索結果(似ている順)---")
for i, idx in enumerate(indices[0]):
    print(f"\n【{i+1}位】(距離: {distances[0][i]:.2f})")
    print(chunks[idx])