# main.py
import os
import uvicorn
import json
import torch
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
app = FastAPI()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2. LOAD MODEL & DATA
print("Memuat model SBERT 'distiluse-base-multilingual-cased-v1'...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

print("Memuat database karier dari careers.json...")
with open('careers.json', 'r', encoding='utf-8') as f:
    careers_data = json.load(f)

career_descriptions = [career['description'] for career in careers_data]
print(f"Berhasil memuat {len(career_descriptions)} deskripsi karier.")
print("Membuat embedding karier...")
emb_careers = model.encode(career_descriptions, convert_to_tensor=True)
print("Startup selesai. Menunggu input pengguna...")

# 3. DEFINISI ENDPOINT
class UserInput(BaseModel):
    text: str

@app.post("/process")
async def process_text(user_input: UserInput):
    print("="*60)
    print(f"[INPUT PENGGUNA] {user_input.text}")

    # A. Hitung similarity dengan SBERT
    emb_user = model.encode(user_input.text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_user, emb_careers)[0]

    # Ambil top 3 karier berdasarkan skor tertinggi
    top_indices = torch.topk(cosine_scores, k=3).indices.tolist()
    top_matches = [careers_data[i] for i in top_indices]

    print("[TOP-3 HASIL SBERT]")
    for i, career in enumerate(top_matches):
        print(f"{i+1}. {career['title']} (Skor: {round(float(cosine_scores[top_indices[i]]), 2)})")

    # B. Siapkan prompt Gemini
    llm_model = genai.GenerativeModel('gemini-1.5-flash')

    # --- SEMUA BLOK DI BAWAH INI SUDAH DIBERI INDENTASI YANG BENAR ---

    # --- Persiapan Data untuk Prompt ---
    best_match_career = top_matches[0]
    best_match_title = best_match_career['title']
    best_match_description = best_match_career['description']

    # Ambil 2 karier lainnya sebagai alternatif
    other_options = [match['title'] for match in top_matches[1:]]
    other_options_string = " dan ".join(other_options)

    # --- PROMPT BARU YANG DISEMPURNAKAN ---
    prompt = f"""
## Persona
Anda adalah "AI Buddy", seorang asisten karier yang hangat, cerdas, dan suportif. Tujuan utamamu adalah membuat pengguna merasa didengar, dipahami, dan termotivasi. Gunakan gaya bahasa yang alami seperti sedang mengobrol.

## Konteks
Seorang pengguna baru saja bercerita tentang minatnya.
- Cerita Pengguna: "{user_input.text}"
- Analisis sistem menunjukkan kecocokan terkuat dengan profesi: "{best_match_title}".
- Deskripsi profesi ini: "{best_match_description}"
- Opsi menarik lainnya adalah: "{other_options_string}".

## Tugas Utama
Buat sebuah respons percakapan yang singkat, padat, dan jelas. Jangan terdengar seperti robot yang menjawab instruksi. Respons harus mengalir secara alami.

## Alur Respon yang Diinginkan
1.  **Sapaan & Pengakuan:** Mulai dengan kalimat yang mengakui dan merespons langsung cerita pengguna.
2.  **Fokus pada Rekomendasi Utama:** Sebutkan bahwa ceritanya sangat cocok dengan profesi "{best_match_title}". Jelaskan alasannya secara singkat dengan menyadur inti dari deskripsi yang ada. Hubungkan langsung dengan apa yang dikatakan pengguna.
3.  **Sebutkan Alternatif (secara natural):** Secara sambil lalu, sebutkan bahwa ada juga bidang lain yang menarik untuk dieksplorasi seperti "{other_options_string}".
4.  **Penutup yang Memotivasi:** Akhiri dengan kalimat penyemangat.

## Batasan
- Jawaban harus dalam bentuk teks biasa (plain text).
- Jangan gunakan format markdown (simbol *, #, -, dll.).
- Jaga agar respons tetap singkat, sekitar 3-4 kalimat.

Contoh output yang baik:
"Wow, dari ceritamu sepertinya kamu punya potensi besar sebagai Software Engineer! Kemampuanmu dalam logika dan memecahkan masalah sangat cocok dengan profesi ini. Selain itu, bidang lain seperti UI/UX Designer dan Data Scientist juga bisa kamu eksplorasi. Terus kembangkan minatmu ya!"
"""

    try:
        response = await llm_model.generate_content_async(prompt)
        final_response_text = response.text
        print("\n[RESPONS GEMINI]")
        print(final_response_text)
    except Exception as e:
        print(f"[ERROR Gemini] {e}")
        final_response_text = "Maaf, AI sedang tidak bisa memberikan rekomendasi. Coba lagi sebentar ya."

    print("="*60 + "\n")

    return {
        "reply": final_response_text,
        "shortlist_titles": [match['title'] for match in top_matches]
    }

# 4. RUN
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
