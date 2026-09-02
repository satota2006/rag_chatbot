import pdfplumber
from sentence_transformers import SentenceTransformer

pdf_path = "sample.pdf"  # 自分のPDFファイル名に変更してください

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

# 埋め込みモデルを読み込む(日本語にも対応したモデルを使用)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 全チャンクをベクトル化
embeddings = model.encode(chunks)

print(f"チャンク数: {len(chunks)}")
print(f"ベクトルの形: {embeddings.shape}")
print("---最初のチャンクのベクトル(最初の10個の数字だけ表示)---")
print(embeddings[0][:10])