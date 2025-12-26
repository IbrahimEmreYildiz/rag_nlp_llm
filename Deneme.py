from google.genai import Client
import os

from google.genai import Client

client = Client(api_key="AIzaSyB2yw3VZagtmSerEiOE_fcD0ppSTikeUOo")


# Mevcut modelleri listeleme
models = client.list_models()  # artık burada çalışmalı
for m in models:
    print(m.name, m.capabilities)
