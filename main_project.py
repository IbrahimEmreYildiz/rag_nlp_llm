import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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

    # 1. LLM (Beyin) Kurulumu
    # NOT: Buraya kendi Gemini API anahtarını koymalısın
    os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # 2. Prompt (Talimat) Hazırlama
    # Modele nasıl davranması gerektiğini söylüyoruz
    sablon = """
    Sen akademik bir asistansın. Aşağıdaki bağlam (context) bilgilerini kullanarak soruyu cevapla.
    Eğer cevap bağlamda yoksa 'Bu bilgi dökümanda bulunmuyor' de, uydurma.

    BAĞLAM:
    {context}

    SORU: {question}

    CEVAP:"""

    PROMPT = PromptTemplate(template=sablon, input_variables=["context", "question"])

    # 3. RAG Zincirini Oluşturma
    rag_zinciri = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    # 4. Final Test
    soru = "What are the three types of paraphrases mentioned in the document?"
    print(f"\nAsistan Yanıtlıyor...")
    cevap = rag_zinciri.invoke(soru)
    print(cevap["result"])