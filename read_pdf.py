import pdfplumber

# ここを、実際のPDFファイル名に書き換えてください
pdf_path = "sample.pdf"

with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

print(all_text)