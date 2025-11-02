import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

oai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="AI Chatbot", layout="wide")

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

    .budget-section {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .budget-row {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .cost-display {
        font-size: 1.1rem;
        font-weight: bold;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found in .env")
if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY not found in .env")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "cost" not in st.session_state:
    st.session_state.cost = 0.0

st.sidebar.markdown('<p class="sidebar-title">Settings</p>', unsafe_allow_html=True)

provider = st.sidebar.selectbox("Choose Provider", ["OpenAI", "Gemini"])

if provider == "OpenAI":
    model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
else:
    model = st.sidebar.selectbox("Model", [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest"
    ])

st.sidebar.markdown("---")

with st.sidebar.expander("Advanced Settings", expanded=True):
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.95, 0.05)
    top_k = st.slider("Top-k", 1, 100, 40, 1)
    max_tokens = st.number_input("Max Tokens", 50, 2048, 512, 64)

st.sidebar.markdown("---")
min_budget = 1.0
max_budget = 5.0
current_cost = st.session_state.cost
cost_color = "#00ff00" if current_cost <= max_budget else "#ff0000"

budget_html = f"""
<div class="budget-section">
    <div style="font-weight: 600; margin-bottom: 0.5rem;">Budget Control</div>
    <div class="budget-row">
        <span>Min Budget:</span>
        <span>${min_budget:.2f}</span>
    </div>
    <div class="budget-row">
        <span>Max Budget:</span>
        <span>${max_budget:.2f}</span>
    </div>
    <div class="cost-display" style="color: {cost_color};">
        Current: ${current_cost:.4f}
    </div>
</div>
"""
st.sidebar.markdown(budget_html, unsafe_allow_html=True)

st.sidebar.markdown("---")

if st.sidebar.button("Summarize Chat"):
    if not st.session_state.messages:
        st.sidebar.info("No messages to summarize")
    else:
        transcript = "\n".join(
            [f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
             for m in st.session_state.messages]
        )
        prompt = "Summarize the following conversation in 5 concise bullet points:\n\n" + transcript

        with st.spinner("Generating summary..."):
            summary = None
            if provider == "OpenAI" and oai_client:
                try:
                    resp = oai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        max_tokens=300,
                    )
                    summary = resp.choices[0].message.content.strip()
                except Exception as e:
                    summary = f"Error: {e}"
            elif provider == "Gemini" and GEMINI_API_KEY:
                try:
                    gmodel = genai.GenerativeModel(model)
                    cfg = genai.GenerationConfig(temperature=0.5, max_output_tokens=300)
                    resp = gmodel.generate_content(prompt, generation_config=cfg)
                    summary = resp.text.strip()
                except Exception as e:
                    summary = f"Error: {e}"

            if summary:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"**Summary:**\n\n{summary}"}
                )
                st.rerun()

def to_gemini_history(messages):
    history = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["content"]]})
    return history

def estimate_cost_openai(tokens: int) -> float:
    return tokens * 0.000005

def call_model(messages, user_prompt):
    if provider == "OpenAI":
        if not oai_client:
            return "ERROR: OPENAI_API_KEY not configured"
        try:
            resp = oai_client.chat.completions.create(
                model=model,
                messages=messages + [{"role": "user", "content": user_prompt}],
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )
            reply = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            if usage and getattr(usage, "total_tokens", None) is not None:
                st.session_state.cost += estimate_cost_openai(usage.total_tokens)
            return reply
        except Exception as e:
            return f"ERROR: {e}"
    else:
        if not GEMINI_API_KEY:
            return "ERROR: GEMINI_API_KEY not configured"
        try:
            gmodel = genai.GenerativeModel(model)

            full_prompt = ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                full_prompt += f"{role}: {msg['content']}\n\n"
            full_prompt += f"User: {user_prompt}\n\nAssistant:"

            cfg = genai.GenerationConfig(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                max_output_tokens=int(max_tokens),
            )

            resp = gmodel.generate_content(full_prompt, generation_config=cfg)
            return resp.text.strip()
        except Exception as e:
            return f"ERROR: {e}"

st.title("AI Chatbot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = call_model(st.session_state.messages[:-1], user_input)
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
