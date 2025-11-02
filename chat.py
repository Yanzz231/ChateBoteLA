import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Local Chatbot", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }

    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }

    .stButton button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 4px;
    }

    .stButton button:hover {
        background-color: #0052a3;
    }

    .info-section {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-title">Settings</p>', unsafe_allow_html=True)

model_options = ["llama3.2", "deepseek-r1:1.5b", "phi3:mini"]
MODEL = st.sidebar.selectbox("Choose a Model", model_options, index=0)

MAX_HISTORY = st.sidebar.number_input("Max History", 1, 10, 2, 1)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", 1024, 16384, 8192, 1024)

st.sidebar.markdown("---")

with st.sidebar.expander("Advanced Settings", expanded=True):
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.95, 0.05)
    top_k = st.slider("Top-k", 1, 100, 40, 1)
    max_tokens = st.number_input("Max Tokens", 50, 4096, 2048, 128)

st.sidebar.markdown("---")
info_html = f"""
<div class="info-section">
    <div style="font-weight: 600; margin-bottom: 0.5rem;">Connection Info</div>
    <div class="info-row">
        <span>Status:</span>
        <span>Local</span>
    </div>
    <div class="info-row">
        <span>Model:</span>
        <span>{MODEL}</span>
    </div>
    <div class="info-row">
        <span>Context:</span>
        <span>{CONTEXT_SIZE}</span>
    </div>
    <div class="info-row">
        <span>History:</span>
        <span>{MAX_HISTORY}</span>
    </div>
</div>
"""
st.sidebar.markdown(info_html, unsafe_allow_html=True)

st.sidebar.markdown("---")

if st.sidebar.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

def clear_memory():
    st.session_state.chat_history = []

if "prev_context_size" not in st.session_state or st.session_state.prev_context_size != CONTEXT_SIZE:
    clear_memory()
    st.session_state.prev_context_size = CONTEXT_SIZE

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

try:
    llm = ChatOllama(
        model=MODEL,
        streaming=True,
        num_ctx=CONTEXT_SIZE,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=max_tokens
    )
    ollama_available = True
except Exception:
    ollama_available = False

st.title("Local Chatbot")

if not ollama_available:
    st.error("Ollama not running. Please install and start Ollama, then refresh this page.")
    st.info("""
    **Setup Instructions:**
    1. Download from: https://ollama.ai
    2. Run: `ollama serve`
    3. Pull model: `ollama pull llama3.2`
    4. Refresh this page
    """)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:
        st.session_state.chat_history.pop(0)
        if st.session_state.chat_history:
            st.session_state.chat_history.pop(0)

if ollama_available:
    if prompt := st.chat_input("Type your message..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        trim_memory()

        messages = []
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            try:
                for chunk in llm.stream(messages):
                    if hasattr(chunk, 'content'):
                        full_response += chunk.content
                        response_container.markdown(full_response)
            except Exception as e:
                full_response = f"ERROR: {str(e)}"
                response_container.markdown(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        trim_memory()
