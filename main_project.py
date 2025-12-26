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
print(os.getenv("GOOGLE_API_KEY"))



# --- AYARLAR VE DOSYA YOLLARI ---
PDF_YOLU = "data/NLP13.pdf"
DB_YOLU = "./chroma_db_deposu2"

# 1. VERİ YÜKLEME VE ÖN İŞLEME
if not os.path.exists(DB_YOLU):
    print("--- Yeni Veritabanı Oluşturuluyor ---")

    # PDF'i oku
    loader = PyPDFLoader(PDF_YOLU)
    belge = loader.load()

    # Metni parçalara böl (800 karakter boyut, 150 karakter bindirme)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    parcalar = text_splitter.split_documents(belge)
    print(f"PDF parçalandı: {len(parcalar)} adet parça oluştu.")

    # 2. VEKTÖRLEŞTİRME VE KAYIT
    # all-mpnet-base-v2 modelini kullanarak sayısal vektörler üretir
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    vector_db = Chroma.from_documents(
        documents=parcalar,
        embedding=embedding_model,
        persist_directory=DB_YOLU
    )
    print("Vektörleştirme bitti ve ChromaDB'ye kaydedildi.")

else:
    # Veritabanı zaten varsa doğrudan oradan oku (Zaman tasarrufu)
    print("--- Mevcut Veritabanı Yükleniyor ---")
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vector_db = Chroma(persist_directory=DB_YOLU, embedding_function=embedding_model)

# 3. ARAMA (RETRIEVAL) TESTİ
soru = "What is the definition of a paraphrase according to the document?"
print(f"\nSoru: {soru}")

# En alakalı 3 parçayı getir
bulunan_parcalar = vector_db.similarity_search(soru, k=3)

print("\n--- Bulunan Kaynaklar ---")
for i, parca in enumerate(bulunan_parcalar):
    # Sayfa numarasını 0-indeksinden 1-indeksine çevirerek düzeltiyoruz
    sayfa_no = parca.metadata.get('page', 0) + 1
    print(f"[{i + 1}] Sayfa {sayfa_no}: {parca.page_content[:150]}...")


# 1. LLM Kurulumu
# main_project.py içindeki llm satırını şu şekilde güncelle:
# Sadece bu satırı değiştir:
# Sadece bu satırı güncelle:
# Sadece bu satırı değiştirip dene:
# Eski satırı sil ve bunu yapıştır:
# Test çıktısına göre ismi tam olarak şöyle yazmayı dene:
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    temperature=0.3,
)


template = """Aşağıdaki bağlamı kullanarak soruyu cevapla:
{context}

Soru: {question}
Cevap:"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": vector_db.as_retriever() | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

soru = "What is the definition of a paraphrase?"
cevap = rag_chain.invoke(soru)
print("\n--- MODEL CEVABI ---")
print(cevap)
