from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
# print(os.getenv("GOOGLE_API_KEY")) # Güvenlik için kapattım

# AYARLAR VE DOSYA YOLLARI
PDF_YOLU = "data/NLP13.pdf"
DB_YOLU = "./chroma_db_deposu2"

# 1. VERİ YÜKLEME VE ÖN İŞLEME
if not os.path.exists(DB_YOLU):
    print("--- Yeni Veritabanı Oluşturuluyor ---")

    # PDF'i oku
    loader = PyPDFLoader(PDF_YOLU)
    belge = loader.load()

    # Metni parçalara böl
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    parcalar = text_splitter.split_documents(belge)
    print(f"PDF parçalandı: {len(parcalar)} adet parça oluştu.")

    # 2. VEKTÖRLEŞTİRME VE KAYIT
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    vector_db = Chroma.from_documents(
        documents=parcalar,
        embedding=embedding_model,
        persist_directory=DB_YOLU
    )
    print("Vektörleştirme bitti ve ChromaDB'ye kaydedildi.")

else:
    # Veritabanı zaten varsa oradan oku
    print("--- Mevcut Veritabanı Yükleniyor ---")
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vector_db = Chroma(persist_directory=DB_YOLU, embedding_function=embedding_model)


# 3. LLM AYARLARI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# Asistana dil kuralını burada öğretiyoruz
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

# Zinciri oluşturuyoruz
rag_chain = (
    {
        "context": vector_db.as_retriever() | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- ÇALIŞTIRMA DÖNGÜSÜ ---
print("\n--- RAG Asistanı Hazır! (Çıkmak için 'q' yazın) ---\n")

while True:
    kullanici_sorusu = input("Sorunuz: ")

    if kullanici_sorusu.lower() in ['q', 'exit', 'çık']:
        print("Görüşürüz! 👋")
        break

    if not kullanici_sorusu.strip():
        continue

    print("🤖 Düşünüyor...")

    try:
        cevap = rag_chain.invoke(kullanici_sorusu)
        print(f"\nCevap:\n{cevap}\n")
        print("-" * 50)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")