# AI Chatbot Assignment

Dua aplikasi chatbot berbasis Streamlit:
- **app.py** - Chatbot dengan OpenAI dan Google Gemini
- **chat.py** - Chatbot lokal dengan Ollama

## Instalasi

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup untuk app.py (OpenAI/Gemini)

**a. Copy file `.env.example` menjadi `.env`:**

```bash
# Windows
copy ".env copy.example" .env

# Linux/Mac
cp .env.example .env
```

**b. Edit file `.env` dan masukkan API keys:**

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Dapatkan API Keys dari:
- OpenAI: https://platform.openai.com/api-keys
- Gemini: https://makersuite.google.com/app/apikey

### 3. Setup untuk chat.py (Ollama) - Opsional

**a. Install Ollama:**
- Download: https://ollama.ai

**b. Pull model:**

```bash
ollama pull llama3.2
ollama pull deepseek-r1:1.5b
```

**c. Jalankan Ollama:**

```bash
ollama serve
```

## Cara Menjalankan

### Chatbot Cloud (app.py)

```bash
python -m streamlit run app.py
```

Buka browser di `http://localhost:8501`

### Chatbot Lokal (chat.py)

```bash
python -m streamlit run chat.py
```

Buka browser di `http://localhost:8501`

**Catatan:** Jika command `streamlit` tidak ditemukan, gunakan `python -m streamlit` seperti contoh di atas.
