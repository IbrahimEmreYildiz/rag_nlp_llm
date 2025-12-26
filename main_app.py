import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. AYARLAR VE YAPILANDIRMA
# ---------------------------------------------------------
st.set_page_config(page_title="RAG Asistanı", page_icon="🤖")
st.title("📄 PDF Sohbet Asistanı")
st.markdown("*Ders notlarınızla konuşun...*")

load_dotenv()

PDF_YOLU = "data/NLP13.pdf"
DB_YOLU = "./chroma_db_deposu2"


# ---------------------------------------------------------
# 2. VERİTABANI OLUŞTURMA (CACHING İLE)
# ---------------------------------------------------------
@st.cache_resource
def get_retriever():
    with st.spinner("Yapay zeka modelleri ve veritabanı yükleniyor... (İlk açılış 1-2 dk sürebilir)"):
        if not os.path.exists(DB_YOLU):
            loader = PyPDFLoader(PDF_YOLU)
            belge = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, length_function=len)
            parcalar = text_splitter.split_documents(belge)
            embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            vector_db = Chroma.from_documents(documents=parcalar, embedding=embedding_model, persist_directory=DB_YOLU)
        else:
            embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            vector_db = Chroma(persist_directory=DB_YOLU, embedding_function=embedding_model)
        # k=6 diyerek okuma kapasitesini artırıyoruz (Eskiden 4'tü)
            # k=10 yaparak daha geniş bir alanı taramasını sağlıyoruz
            return vector_db.as_retriever(search_kwargs={"k": 10})


# ---------------------------------------------------------
# 3. YAPAY ZEKA ZİNCİRİNİ (CHAIN) KURMA
# ---------------------------------------------------------
def get_chain(retriever):
    # ✅ DÜZELTİLEN KISIM: ChatGoogleGenerativeAI ismi doğru
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # EKSİK OLAN KISIMLAR BURADAN İTİBAREN BAŞLIYOR:
    template = """
    Sen yardımsever bir asistansın. Aşağıdaki bağlamı (context) kullanarak soruyu cevapla.

    KURALLAR:
    1. Eğer soru Türkçe ise cevabı TÜRKÇE ver.
    2. Eğer soru İngilizce ise cevabı İNGİLİZCE ver.
    3. Bağlam (context) İngilizce olsa bile, sen her zaman SORUNUN DİLİNDE cevap ver.

    Bağlam:
    {context}

    Soru: {question}

    Cevap:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


# ---------------------------------------------------------
# 4. ARAYÜZ MANTIĞI (SENİN KODUNDA EKSİK OLAN KISIM)
# ---------------------------------------------------------

# Sistemin hazır olup olmadığını kontrol et
try:
    retriever = get_retriever()
    chain = get_chain(retriever)
    st.success("Sistem Hazır! ✅ Sohbet edebilirsiniz.")
except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
    st.stop()

# Sohbet geçmişini hafızada tut
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan girdi al
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan cevabını üret
    with st.chat_message("assistant"):
        with st.spinner("Cevap oluşturuluyor..."):
            response = chain.invoke(prompt)
            st.markdown(response)

    # Asistan mesajını kaydet
    st.session_state.messages.append({"role": "assistant", "content": response})