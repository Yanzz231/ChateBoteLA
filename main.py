import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from datetime import datetime

st.set_page_config(layout="wide")
st.title("My Local Chatbot")
st.sidebar.header("Settings")

model_options = ["mistral", "phi3.5", "qwen2.5:4b"]
MODEL = st.sidebar.selectbox("Choose a Model", model_options, index=0)

# Advanced settings (temperature / sampling / max tokens)
with st.sidebar.expander("Advanced settings", expanded=False):
    TEMPERATURE = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, help="Higher = more creative")
    TOP_P = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05, help="Nucleus sampling")
    TOP_K = st.slider("Top-k", 1, 200, 40, 1, help="Sample from top-k tokens")
    MAX_TOKENS = st.slider("Max tokens", 32, 4096, 512, 32, help="Maximum new tokens to generate")

with st.sidebar.expander("Summary settings", expanded=False):
    SUMMARY_MAX_BULLETS = st.slider("Max bullets", 3, 10, 5, 1)
    SUMMARY_MAX_TOKENS = st.slider("Summary max tokens", 64, 512, 192, 32)
    SUMMARY_STYLE = st.radio("Summary style", ["Auto", "Paragraph", "Bullets"], index=0, horizontal=True)

# Prompt Engineering Section
with st.sidebar.expander("Prompt Engineering", expanded=False):
    st.caption("Customize how the AI responds")

    # Prompt Templates
    PROMPT_TEMPLATES = {
        "Default": "You are a helpful AI assistant. Keep answers concise unless asked for detail.",
        "Professional": "You are a professional AI assistant. Provide clear, well-structured, and formal responses. Use proper terminology and maintain a business-appropriate tone.",
        "Creative": "You are a creative and imaginative AI assistant. Think outside the box, use vivid language, and provide unique perspectives. Feel free to be expressive and engaging.",
        "Technical Expert": "You are a technical expert AI assistant. Provide detailed, accurate technical information with examples. Use precise terminology and explain complex concepts clearly.",
        "Concise": "You are a concise AI assistant. Provide brief, direct answers. Get straight to the point without unnecessary elaboration.",
        "Friendly Tutor": "You are a friendly and patient tutor. Explain concepts in simple terms, use analogies, and encourage learning. Break down complex topics into easy-to-understand parts.",
        "Analytical": "You are an analytical AI assistant. Provide logical, data-driven responses. Break down problems systematically and consider multiple perspectives.",
        "Code Expert": "You are an expert programming assistant. Provide clean, efficient code with explanations. Follow best practices and include comments where helpful.",
        "Custom": ""
    }

    prompt_choice = st.selectbox(
        "Select Prompt Template",
        options=list(PROMPT_TEMPLATES.keys()),
        index=0,
        help="Choose a pre-built template or create your own"
    )

    # Show custom text area if Custom is selected
    if prompt_choice == "Custom":
        SYSTEM_PROMPT = st.text_area(
            "Custom System Prompt",
            value="You are a helpful AI assistant.",
            height=100,
            help="Write your own system prompt to guide the AI's behavior"
        )
    else:
        SYSTEM_PROMPT = PROMPT_TEMPLATES[prompt_choice]
        st.info(f"**Current prompt:** {SYSTEM_PROMPT[:100]}...")

    # Additional prompt modifiers
    st.caption("Prompt Modifiers")

    col1, col2 = st.columns(2)
    with col1:
        ADD_ROLE = st.checkbox("Add Role", value=False, help="Add a specific role/persona")
    with col2:
        ADD_FORMAT = st.checkbox("Format Output", value=False, help="Specify output format")

    ROLE_CONTEXT = ""
    FORMAT_CONTEXT = ""

    if ADD_ROLE:
        role_options = [
            "Software Engineer",
            "Data Scientist",
            "Teacher",
            "Business Analyst",
            "Creative Writer",
            "Research Assistant",
            "Custom Role"
        ]
        selected_role = st.selectbox("Role", role_options)
        if selected_role == "Custom Role":
            ROLE_CONTEXT = st.text_input("Enter custom role:", "")
        else:
            ROLE_CONTEXT = selected_role

    if ADD_FORMAT:
        format_options = [
            "Markdown with headers",
            "Numbered list",
            "Bullet points",
            "Step-by-step guide",
            "Q&A format",
            "Custom format"
        ]
        selected_format = st.selectbox("Output Format", format_options)
        if selected_format == "Custom format":
            FORMAT_CONTEXT = st.text_input("Specify format:", "")
        else:
            FORMAT_CONTEXT = selected_format

# Build final system prompt with modifiers
FINAL_SYSTEM_PROMPT = SYSTEM_PROMPT
if ADD_ROLE and ROLE_CONTEXT:
    FINAL_SYSTEM_PROMPT = f"You are a {ROLE_CONTEXT}. {FINAL_SYSTEM_PROMPT}"
if ADD_FORMAT and FORMAT_CONTEXT:
    FINAL_SYSTEM_PROMPT = f"{FINAL_SYSTEM_PROMPT} Format your responses using: {FORMAT_CONTEXT}."

# Inputs for max history and context size
MAX_HISTORY = st.sidebar.number_input("Max History", min_value=1, max_value=10, value=2, step=1)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", min_value=1024, max_value=16384, value=4096, step=1024)

def clear_memory():
    st.session_state.chat_history = []
    st.session_state.memory = ChatMessageHistory()

def ensure_history_store():
    if "history_sessions" not in st.session_state:
        st.session_state.history_sessions = []

def archive_current_session(reason: str = "settings-change"):
    """Save the current chat to the in-session history list (ephemeral)."""
    ensure_history_store()
    if st.session_state.get("chat_history"):
        st.session_state.history_sessions.append({
            "model": st.session_state.get("prev_model", MODEL),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "messages": list(st.session_state.chat_history),
        })

if "prev_context_size" not in st.session_state or st.session_state.prev_context_size != CONTEXT_SIZE:
    archive_current_session("context-size-change")
    clear_memory()
    st.session_state.prev_context_size = CONTEXT_SIZE

# Reset model settings if any of the advanced settings change
if (
    "prev_model" not in st.session_state
    or st.session_state.prev_model != MODEL
    or "prev_temperature" not in st.session_state
    or st.session_state.prev_temperature != TEMPERATURE
    or "prev_top_p" not in st.session_state
    or st.session_state.prev_top_p != TOP_P
    or "prev_top_k" not in st.session_state
    or st.session_state.prev_top_k != TOP_K
    or "prev_max_tokens" not in st.session_state
    or st.session_state.prev_max_tokens != MAX_TOKENS
    or "prev_system_prompt" not in st.session_state
    or st.session_state.prev_system_prompt != FINAL_SYSTEM_PROMPT
):
    archive_current_session("model-or-params-change")
    clear_memory()
    st.session_state.prev_model = MODEL
    st.session_state.prev_temperature = TEMPERATURE
    st.session_state.prev_top_p = TOP_P
    st.session_state.prev_top_k = TOP_K
    st.session_state.prev_max_tokens = MAX_TOKENS
    st.session_state.prev_system_prompt = FINAL_SYSTEM_PROMPT

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()

model = ChatOllama(
    model=MODEL,
    streaming=True,
    num_ctx=CONTEXT_SIZE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    num_predict=MAX_TOKENS,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", FINAL_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])
chain = prompt | model | StrOutputParser()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sidebar: History controls
ensure_history_store()
with st.sidebar.expander("History", expanded=False):
    st.caption(f"Saved this session: {len(st.session_state.history_sessions)} chats")
    if st.button("New Chat", use_container_width=True):
        archive_current_session("manual-new-chat")
        clear_memory()
    if st.session_state.history_sessions:
        labels = [f"{idx+1}. {s['model']} — {s['created_at']} ({len(s['messages'])} msgs)" for idx, s in enumerate(st.session_state.history_sessions)]
        sel = st.selectbox("Past chats", options=list(range(len(labels))), format_func=lambda i: labels[i])
        if sel is not None:
            transcript = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.history_sessions[sel]["messages"]])
            st.text_area("Preview", transcript, height=150)
        if st.button("Clear history"):
            st.session_state.history_sessions = []

def summarize_conversation():
    if not st.session_state.chat_history:
        st.sidebar.info("No messages to summarize yet.")
        return
    # Compose a text transcript from our UI history
    transcript = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.chat_history]
    )
    # Dynamic length budget: about half of the chat length, within bounds
    char_limit = min(800, max(200, int(len(transcript) * 0.5)))

    sum_model = ChatOllama(
        model=MODEL,
        streaming=False,
        num_ctx=CONTEXT_SIZE,
        temperature=min(0.7, TEMPERATURE),
        top_p=TOP_P,
        top_k=TOP_K,
        num_predict=SUMMARY_MAX_TOKENS,
    )
    # Style-aware prompt
    if SUMMARY_STYLE == "Paragraph":
        sys_msg = (
            f"Summarize the conversation as a single short paragraph under {char_limit} characters. "
            "No bullet points. Be concise and avoid repetition. Do not include word/character counts or length statements."
        )
    elif SUMMARY_STYLE == "Bullets":
        sys_msg = (
            f"Summarize the conversation as at most {SUMMARY_MAX_BULLETS} concise bullet points under {char_limit} characters. "
            "One line per bullet. Avoid filler. Do not include word/character counts or length statements."
        )
    else:
        sys_msg = (
            f"Write a concise summary under {char_limit} characters. Prefer a short paragraph for simple topics; "
            f"use up to {SUMMARY_MAX_BULLETS} bullets only if it improves clarity. Avoid repetition. "
            "Do not include word/character counts or length statements."
        )

    sum_prompt = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("human", "{conversation}")
    ])
    sum_chain = sum_prompt | sum_model | StrOutputParser()
    with st.spinner("Summarizing chat…"):
        summary = sum_chain.invoke({"conversation": transcript})

    # Remove any accidental word/character count lines from model output
    def _strip_meta(text: str) -> str:
        lines = []
        for line in text.splitlines():
            lower = line.strip().lower()
            if (
                "word count" in lower or
                "character count" in lower or
                "characters:" in lower or
                "words:" in lower or
                "under the character limit" in lower
            ):
                continue
            lines.append(line)
        return "\n".join(lines).strip()
    summary = _strip_meta(summary)

    # If the summary is too long, do a concise rewrite pass respecting the chosen style
    if len(summary) >= len(transcript) or len(summary) > char_limit:
        if SUMMARY_STYLE == "Paragraph":
            rewrite_sys = f"Rewrite as a single short paragraph under {char_limit} characters. No bullets."
        elif SUMMARY_STYLE == "Bullets":
            rewrite_sys = f"Rewrite as at most {SUMMARY_MAX_BULLETS} one-line bullet points under {char_limit} characters."
        else:
            rewrite_sys = (
                f"Rewrite to be under {char_limit} characters. Prefer a short paragraph; use up to {SUMMARY_MAX_BULLETS} bullets only if clearer."
            )
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", rewrite_sys),
            ("human", "{draft}")
        ])
        rewrite_chain = rewrite_prompt | sum_model | StrOutputParser()
        summary = _strip_meta(rewrite_chain.invoke({"draft": summary}))
    st.sidebar.markdown("---")
    st.sidebar.subheader("Chat Summary")
    st.sidebar.markdown(summary)

if st.sidebar.button("Summarize Chat", use_container_width=True):
    summarize_conversation()

# Display active system prompt
with st.sidebar.expander("Active System Prompt", expanded=False):
    st.code(FINAL_SYSTEM_PROMPT, language=None)
    st.caption(f"Length: {len(FINAL_SYSTEM_PROMPT)} characters")

def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:  # Each cycle has 2 messages (User + AI)
        st.session_state.chat_history.pop(0)  # Remove oldest User message
        if st.session_state.chat_history:
            st.session_state.chat_history.pop(0)  # Remove oldest AI response

if user_input := st.chat_input("Say something"):
    # Show User Input Immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({"role": "user", "content": user_input})  # Store user input

    trim_memory()

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        lc_history = st.session_state.memory.messages

        # Stream response from the chain
        for chunk in chain.stream({"history": lc_history, "input": user_input}):
            if chunk:
                full_response += chunk
                response_container.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    st.session_state.memory.add_user_message(user_input)
    st.session_state.memory.add_ai_message(full_response)

    trim_memory()
