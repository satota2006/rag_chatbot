import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

st.title("📚 授業資料チャットボット")

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

def split_into_chunks(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# チャット履歴を保存する場所を用意(まだ無ければ作る)
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("授業資料(PDF)をアップロードしてください", type="pdf")

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        all_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    chunks = split_into_chunks(all_text)
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    st.success(f"読み込み完了！({len(chunks)}個のチャンクに分割されました)")

    # これまでのチャット履歴を画面に表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # チャット入力欄(Enterで送信できるタイプ)
    query = st.chat_input("質問を入力してください")

    if query:
        # ユーザーの質問を履歴に追加して表示
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # 検索
        query_vector = model.encode([query])
        k = 3
        distances, indices = index.search(np.array(query_vector).astype('float32'), k)
        context = "\n\n".join([chunks[idx] for idx in indices[0]])

        prompt = f"""以下の資料の内容をもとに、質問に日本語で分かりやすく答えてください。

資料:
{context}

質問: {query}

回答:"""

        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                response = ollama.chat(model='llama3', messages=[
                    {'role': 'user', 'content': prompt}
                ])
                answer = response['message']['content']
                st.write(answer)

        # AIの回答も履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": answer})