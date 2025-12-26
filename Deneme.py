from google import genai
import os

client = genai.Client(api_key=os.getenv("AIzaSyC7XmS4t4OKa2AfGkhX-xQwCRFzRisSYGE"))

response = client.models.generate_content(
    model="models/gemini-flash-latest",
    contents="Sadece 'çalışıyorum' yaz"
)

print(response.text)
