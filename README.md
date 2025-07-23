# 🤖 AI Recruiter Agent

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)

This intelligent recruiter agent helps streamline your hiring process by leveraging AI to semantically search, analyze, and interact with resumes using natural language. Built with Streamlit, Gemini API, HuggingFace embeddings, and FAISS for blazing-fast vector search.

---

## 🚀 Features

- 🔎 **Smart Resume Search** via semantic vector matching  
- 🧠 **Question Answering (RAG)** using Gemini and LangChain  
- 📂 **PDF Resume Viewer** with download support  
- 🎯 **Filter by Education, Experience, Skills**  
- 💾 **Persistent FAISS Vector Store**  
- ✅ Fully offline resume processing for data security

---

## 🧰 Technologies Used

- **Streamlit** – UI framework  
- **LangChain** – For Gemini integration  
- **Gemini API** – LLM for QA & RAG  
- **FAISS** – Vector DB for semantic search  
- **HuggingFace** – BGE embeddings  
- **PyMuPDF** – PDF parsing  
- **dotenv** – Secret management

---

## 📂 Folder Structure

```
├── app.py
├── data/               # PDF resumes
├── vectorstore/        # FAISS index
├── .env                # GEMINI_API_KEY
```

---

## ⚙️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/ai-recruiter-agent.git
   cd ai-recruiter-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add `.env`:
   ```
   GEMINI_API_KEY=your_gemini_key
   ```

4. Add your resumes (PDF) into `data/` folder.

5. Run:
   ```bash
   streamlit run app.py
   ```

---

## 📜 License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Licensed under the **MIT License**.

---

## 👤 Author

**Prasanna Korada**

---

## 📝 One-Line Summary for README Tools
```
AI recruiter agent for smart resume search and filtering using Gemini, FAISS, and HuggingFace in Streamlit.
```
