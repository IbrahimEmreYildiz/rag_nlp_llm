# 🎓 RAG Tabanlı NLP Ders Asistanı (Lecture Notes Chatbot)

> **Doğal Dil İşleme (NLP)** ders notları ve akademik makalelerle konuşmanızı sağlayan, **Retrieval-Augmented Generation (RAG)** mimarisine sahip yapay zeka asistanı.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-orange)

## 📖 Proje Hakkında

Bu proje, üniversite ders notları (PDF) üzerinde derinlemesine soru-cevap yapabilmek için geliştirilmiştir. Standart anahtar kelime aramasının aksine, bu asistan metnin **anlamsal içeriğini** anlar.

Kullanıcı İngilizce bir akademik makale yüklese bile, asistanla **Türkçe** konuşabilir ve Türkçe cevaplar alabilir. Arka planda **Google Gemini 2.5 Flash** modelinin gücünü ve **LangChain** orkestrasyonunu kullanır.

## 🚀 Temel Özellikler

* **📄 Akıllı Doküman Analizi:** PDF dosyalarını yükler, metni parçalar (Chunking) ve vektörel veriye dönüştürür.
* **🧠 Semantik Arama (Semantic Search):** Kullanıcının sorusuyla en alakalı içeriği anlam (vektör) benzerliğine göre bulur.
* **🌍 Çok Dilli Destek (Cross-Lingual):** Kaynak metin İngilizce olsa bile, sorulara Türkçe cevap verebilir (Prompt Engineering ile özelleştirilmiştir).
* **⚡ Yüksek Performans:** `@st.cache_resource` ile vektör veritabanı bellekte tutulur, her soruda tekrar yükleme yapmaz.
* **💾 Sohbet Hafızası:** Kullanıcı ile olan geçmiş konuşmaları hatırlar ve bağlamı korur.

## 🛠️ Kullanılan Teknolojiler ve Mimari

Bu proje aşağıdaki modern yapay zeka teknolojileri kullanılarak inşa edilmiştir:

* **LLM (Büyük Dil Modeli):** Google Gemini 2.5 Flash
* **Framework:** LangChain (Python)
* **Arayüz (UI):** Streamlit
* **Vektör Veritabanı:** ChromaDB
* **Embedding Modeli:** HuggingFace (`all-mpnet-base-v2`)

### ⚙️ Çalışma Mantığı (Pipeline)

1.  **Ingestion:** PDF dosyası `PyPDFLoader` ile okunur.
2.  **Splitting:** Metin, bağlam kopmaması için `RecursiveCharacterTextSplitter` ile 800 karakterlik parçalara bölünür (150 karakter örtüşmeli).
3.  **Embedding:** Parçalar sayısal vektörlere dönüştürülür ve ChromaDB'ye kaydedilir.
4.  **Retrieval:** Kullanıcı sorusu geldiğinde, en alakalı 10 parça (`k=10`) veritabanından çekilir.
5.  **Generation:** Bulunan parçalar ve soru, Gemini 2.5 modeline gönderilir ve nihai cevap üretilir.

## 💻 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için adımları takip edin:

1.  **Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/kullaniciadi/proje-isminiz.git](https://github.com/kullaniciadi/proje-isminiz.git)
    cd proje-isminiz
    ```

2.  **Sanal ortam oluşturun (Önerilen):**
    ```bash
    python -m venv venv
    # Windows için:
    venv\Scripts\activate
    # Mac/Linux için:
    source venv/bin/activate
    ```

3.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Anahtarını Ayarlayın:**
    Ana dizinde `.env` adında bir dosya oluşturun ve Google Cloud'dan aldığınız API anahtarını ekleyin:
    ```env
    GOOGLE_API_KEY="AIzaSy...SİZİN_ANAHTARINIZ"
    ```

5.  **Uygulamayı Başlatın:**
    ```bash
    streamlit run app.py
    ```

## 📸 Ekran Görüntüleri

*(Buraya Streamlit arayüzünden alacağın 1-2 ekran görüntüsünü ekleyebilirsin. Örneğin, zor bir soruya verdiği cevabın görüntüsü.)*

## 🔮 Gelecek Geliştirmeler

* [ ] Kullanıcının arayüz üzerinden PDF yükleyebilmesi (Dosya yükleme butonu).
* [ ] Farklı LLM modellerinin (OpenAI, Claude) seçilebilmesi.
* [ ] Cevapların kaynak sayfa numaralarının gösterilmesi.

## 📜 Lisans

 MIT Lisansı altındadır.

