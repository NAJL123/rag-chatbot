# 🤖 Local RAG Chatbot: Chat with your PDF

This project is a **Local RAG (Retrieval-Augmented Generation)** system that allows you to chat with any PDF document (like a CV or a Report) privately and for free. It uses **Ollama** to run Large Language Models locally on your machine.

---

## 🚀 Features
- **100% Private:** Your data never leaves your computer.
- **Cost-Free:** No OpenAI API keys needed, thanks to **Ollama**.
- **Fast Response:** Uses a streaming interface for real-time interaction.
- **Smart Memory:** The chatbot remembers previous parts of the conversation.

## 🛠️ Tech Stack
- **Framework:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Local LLMs:** [Ollama](https://ollama.com/) (Models: `phi3:mini` & `nomic-embed-text`)

---

## 📋 Prerequisites

Before running the project, make sure you have:
1. **Python 3.10+** installed.
2. **Ollama** installed and running.
3. Downloaded the necessary models:
   ```bash
   ollama pull phi3:mini
   ollama pull nomic-embed-text