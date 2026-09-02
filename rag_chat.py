import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

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

# FAISSインデックス作成
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

print("準備完了！質問を入力してください（終了するには 'exit' と入力）")

# 質問ループ
while True:
    query = input("\n質問: ")
    if query.lower() == "exit":
        break

    # 質問をベクトル化して検索
    query_vector = model.encode([query])
    k = 3
    distances, indices = index.search(np.array(query_vector).astype('float32'), k)

    # 検索結果のテキストをまとめる
    context = "\n\n".join([chunks[idx] for idx in indices[0]])

    # Ollamaに渡すプロンプトを作成
    prompt = f"""以下の資料の内容をもとに、質問に日本語で分かりやすく答えてください。

資料:
{context}

質問: {query}

回答:"""

    # Ollamaで回答生成
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': prompt}
    ])

    print("\n回答:")
    print(response['message']['content'])