import streamlit as st
import os
import time
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Najlaa's AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0b0c10;
    --surface:   #13151a;
    --border:    #1e2028;
    --accent:    #c8f564;
    --accent2:   #64f5c8;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --human-bg:  #1a1e2a;
    --ai-bg:     #111318;
    --radius:    14px;
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #1a2a0a22, transparent),
                radial-gradient(ellipse 60% 40% at 100% 80%, #0a2a2222, transparent),
                var(--bg) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.5rem; }

/* ── Main content padding ── */
.block-container {
    max-width: 860px !important;
    padding: 2.5rem 2rem 6rem !important;
    margin: 0 auto;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.app-header .logo {
    width: 44px; height: 44px;
    background: var(--accent);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}
.app-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    letter-spacing: -0.03em;
    color: var(--text);
    line-height: 1.1;
}
.app-header .subtitle {
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Status badge ── */
.status-row {
    display: flex; gap: 0.6rem; align-items: center;
    margin-bottom: 2rem; flex-wrap: wrap;
}
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    border: 1px solid;
}
.badge.green  { background: #c8f56415; border-color: #c8f56430; color: var(--accent); }
.badge.blue   { background: #64f5c815; border-color: #64f5c830; color: var(--accent2); }
.badge.gray   { background: #ffffff08; border-color: var(--border); color: var(--muted); }
.badge .dot   { width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
    gap: 0.9rem !important;
}

/* Human bubbles */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stChatMessageContent {
    background: var(--human-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) var(--radius) 4px var(--radius) !important;
    padding: 0.9rem 1.1rem !important;
}

/* AI bubbles */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stChatMessageContent {
    background: var(--ai-bg) !important;
    border: 1px solid #c8f56418 !important;
    border-left: 2px solid var(--accent) !important;
    border-radius: 4px var(--radius) var(--radius) var(--radius) !important;
    padding: 0.9rem 1.1rem !important;
}

.stChatMessageContent p { 
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    color: var(--text) !important;
}

/* Avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    width: 34px !important; height: 34px !important;
    font-size: 0.85rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(10px);
}
[data-testid="stChatInput"]:focus-within {
    border-color: #c8f56450 !important;
    box-shadow: 0 0 0 3px #c8f56410 !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    background: transparent !important;
}
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 8px !important;
    color: #0b0c10 !important;
}

/* ── Sidebar elements ── */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: var(--text);
    font-size: 1rem;
}
[data-testid="stSidebar"] label {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stFileUploadDropzone"] {
    background: var(--bg) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #c8f56450 !important;
}

/* Sliders */
[data-testid="stSlider"] .stSlider > div > div > div {
    background: var(--accent) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    border-color: #c8f56450 !important;
    color: var(--accent) !important;
    background: #c8f56408 !important;
}

/* ── Info / warning ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.4rem !important; color: var(--text) !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: var(--muted);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.empty-state p { font-size: 0.85rem; line-height: 1.6; max-width: 320px; margin: 0 auto; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Toast / success ── */
[data-testid="stSuccess"] {
    background: #c8f56415 !important;
    border: 1px solid #c8f56430 !important;
    color: var(--accent) !important;
    border-radius: var(--radius) !important;
}

/* Sidebar section labels */
.sidebar-section {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin: 1.5rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "phi3:mini"
if "language" not in st.session_state:
    st.session_state.language = "English"

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;">
        <div style="width:36px;height:36px;background:#c8f564;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;font-size:1.1rem;">✦</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:0.95rem;color:#e8eaf0;">Najlaa's Assistant</div>
            <div style="font-size:0.68rem;color:#6b7280;letter-spacing:0.05em;">POWERED BY OLLAMA</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="sidebar-section">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

    # Model settings
    st.markdown('<div class="sidebar-section">Model Settings</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model", ["phi3:mini", "llama3.2", "mistral", "gemma2:2b"],
        index=["phi3:mini", "llama3.2", "mistral", "gemma2:2b"].index(st.session_state.model_choice),
        label_visibility="collapsed"
    )
    st.session_state.model_choice = model_choice

    temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
                            help="Higher = more creative, Lower = more precise")
    st.session_state.temperature = temperature

    # Response language
    st.markdown('<div class="sidebar-section">Response Language</div>', unsafe_allow_html=True)
    language = st.selectbox("Language", ["English", "French", "Arabic", "Spanish", "German"],
                            index=["English", "French", "Arabic", "Spanish", "German"].index(st.session_state.language),
                            label_visibility="collapsed")
    st.session_state.language = language

    # Stats
    if st.session_state.chat_history:
        st.markdown('<div class="sidebar-section">Session Stats</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.chat_history))
        with col2:
            st.metric("Q&A Pairs", len(st.session_state.chat_history) // 2)

    # Clear
    st.markdown('<div class="sidebar-section">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.msg_count = 0
        st.rerun()

    if st.session_state.chat_history:
        # Export chat as text
        chat_export = "\n\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in st.session_state.chat_history
        ])
        st.download_button(
            "⬇  Export conversation",
            data=chat_export,
            file_name="conversation.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("""
    <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #1e2028;
                font-size:0.7rem;color:#4b5563;text-align:center;">
        Built with Streamlit & LangChain
    </div>
    """, unsafe_allow_html=True)

# ─── Main Area ───────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <div class="logo">✦</div>
    <div>
        <h1>AI Document Assistant</h1>
        <div class="subtitle">Ask anything about your PDF</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── RAG Setup ───────────────────────────────────────────────────────────────
if uploaded_file:
    # Save file
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Status badges
    st.markdown(f"""
    <div class="status-row">
        <span class="badge green"><span class="dot"></span> Active</span>
        <span class="badge blue">✦ {st.session_state.model_choice}</span>
        <span class="badge gray">📄 {uploaded_file.name}</span>
        <span class="badge gray">🌐 {st.session_state.language}</span>
        <span class="badge gray">🌡 temp {st.session_state.temperature}</span>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def get_retriever(file_path, model):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return FAISS.from_documents(splits, embeddings).as_retriever(search_kwargs={"k": 4})

    with st.spinner("Indexing document…"):
        retriever = get_retriever(file_path, st.session_state.model_choice)

    llm = ChatOllama(
        model=st.session_state.model_choice,
        temperature=st.session_state.temperature,
        streaming=True
    )

    lang_instruction = f"Always respond in {st.session_state.language}." if st.session_state.language != "English" else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a helpful, precise document assistant. Use the provided context to answer the user's question accurately. "
         f"If the answer isn't in the context, say so honestly. {lang_instruction}\n\nContext:\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def get_history(_):
        return st.session_state.get("chat_history", [])

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": get_history,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # ── Suggested questions (shown only when no history) ──
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:0.8rem;">Suggested questions</div>
        </div>
        """, unsafe_allow_html=True)

        suggestions = [
            "📋 Summarize the main points",
            "🔍 What are the key findings?",
            "📊 List important data or statistics",
            "❓ What is the document about?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, use_container_width=True, key=f"suggest_{i}"):
                user_query = s[2:].strip()  # strip emoji
                st.session_state._pending_query = user_query
                st.rerun()

    # ── Chat history ──
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Handle pending query from suggestion buttons
    pending = getattr(st.session_state, "_pending_query", None)

    # ── Chat input ──
    user_query = st.chat_input("Ask me anything about your document…") or pending
    if pending:
        st.session_state._pending_query = None

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner(""):
                response = st.write_stream(rag_chain.stream(user_query))

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.msg_count += 1

else:
    # Empty state
    st.markdown("""
    <div class="status-row">
        <span class="badge gray">● Waiting for document</span>
    </div>
    <div class="empty-state">
        <div class="icon">📄</div>
        <h3>No document loaded</h3>
        <p>Upload a PDF in the sidebar to start asking questions about its content.</p>
    </div>
    """, unsafe_allow_html=True)