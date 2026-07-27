# 📝 AI Blog Writing Agent

An autonomous, multi-agent technical blog writing system built using **LangGraph**, **Streamlit**, and **Open-Source LLMs (Llama 3.1 8B / Llama 3.3 70B)** via Groq and OpenRouter.

The system performs intelligent routing, real-time web research, structured outline orchestration, parallel map-reduce section writing, and fail-safe AI image generation.

---

## ✨ Features

- **🧭 Intelligent Routing**: Automatically evaluates topic requirements (`closed_book`, `hybrid`, or `open_book` news roundup).
- **🔎 Zero-Token Web Research**: Leverages **Tavily Search API** with direct Python parsing for $0 token cost extraction.
- **📋 Map-Reduce Concurrency**: Writes all blog sections in parallel using LangGraph map-reduce (`Send`).
- **🛡️ Multi-Tier LLM Fallbacks**: Features automatic failover between Groq models and OpenRouter to bypass rate limits seamlessly.
- **🖼️ Fail-Safe Image Generation**: Proposes technical diagrams and places generated images into Markdown. Automatically falls back to **Pollinations AI** if Google Gemini quota is reached.
- **🎨 Interactive Streamlit UI**: Live streaming graph progress, Markdown rendering, past blog library, and ZIP exports.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Agent Orchestration**: LangGraph, LangChain
- **LLM Providers**: Groq API, OpenRouter API
- **Web Search**: Tavily Search API
- **Image Models**: Google Gemini API (`gemini-2.5-flash-image`), Pollinations AI
- **Schema Validation**: Pydantic v2

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/blog-writing-agent.git
cd blog-writing-agent
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

TAVILY_API_KEY=your_tavily_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 💻 Running the Application

Launch the Streamlit web interface:
```bash
streamlit run bwa_frontend.py
```

Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
blog-writing-agent/
├── bwa_backend.py        # LangGraph workflow, nodes, and LLM configuration
├── bwa_frontend.py       # Streamlit user interface & markdown renderer
├── requirements.txt      # Project dependencies
├── .env.example          # Template for environment variables
├── .gitignore            # Git exclusion rules
├── README.md             # Project documentation
└── images/               # Directory for generated blog diagrams
```

---

## 📄 License
MIT License
