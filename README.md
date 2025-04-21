# 🧠 My AI Assistant

A personal AI assistant designed to answer questions about my background, professional experience, opinions, and projects.

## ✨ Features

- Conversational interface powered by LLMs
- Retrieval-Augmented Generation (RAG) with LangChain + Chroma
- Retrieving personal data (resume, projects, interests)
- Deployed with Streamlit for easy access
- Modular structure for adding new data sources
- Designed for real-time interaction

## 🔍 Tech Stack

- **Streamlit** — Web UI
- **LangChain** — Agent and memory framework
- **Chroma** — Vector database for memory retrieval
- **Hugging Face** — Model and embedding source
- **Python** — Core development language

## 🚀 Live Demo

Try it here: [https://adrian-saez-martinez.streamlit.app](https://adrian-saez-martinez.streamlit.app)

## ⚙️ How It Works

1. A user asks a question.
2. The question is embedded and processed by the retriever (LangChain + Chroma).
3. The retriever searches for relevant documents in the Chroma vector DB.
4. The documents are combined with the original question to create a prompt.
5. The prompt is sent to the LLM, which generates an answer.
6. The final response is returned to the user.

## 📌 Future Ideas

- Voice interaction
- Dynamic personality adjustments
- Connecting with external tools (calendar, email, etc.)

## 📬 Contact

Feel free to reach out if you have questions or want to collaborate:

- LinkedIn: [Adrián Sáez Martínez](https://www.linkedin.com/in/adrian-saez-martinez/)
- Email: adriansaezmartinez@email.com

