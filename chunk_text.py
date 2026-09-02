import pdfplumber

pdf_path = "sample.pdf" 
# PDFからテキストを抽出
with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

# テキストをチャンク(小さい塊)に分割する関数
def split_into_chunks(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # 少し重なりを持たせて分割
    return chunks

chunks = split_into_chunks(all_text)

print(f"チャンクの数: {len(chunks)}")
print("---最初のチャンク---")
print(chunks[0])
print("---2番目のチャンク---")
print(chunks[1] if len(chunks) > 1 else "（2番目はありません）")