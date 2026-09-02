import os 
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# On Streamlit Cloud, secrets come from st.secrets (not a .env file).
# This makes the key available locally (.env) AND on Streamlit Cloud (secrets).
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Friendly error instead of a crash if the key is still missing
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "⚠️ GROQ_API_KEY not found.\n\n"
        "If running locally: check your `.env` file has `GROQ_API_KEY=your_key`.\n\n"
        "If deployed on Streamlit Cloud: go to your app's Settings → Secrets, "
        "and make sure it contains exactly:\n\n"
        '```\nGROQ_API_KEY = "your_key_here"\n```\n\n'
        "Then save and reboot the app."
    )
    st.stop()

# ---------- Page settings ----------
st.set_page_config(
    page_title="NotesGenie | AI Study Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS: 3D / glassmorphism look ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b1020;
    --surface: #121a2b;
    --surface-2: #182237;
    --border: #2b3954;
    --text: #eef3ff;
    --muted: #9ba9c3;
    --purple: #8b5cf6;
    --purple-dark: #6d28d9;
}

html, body, [class*="css"] { font-family: "Inter", sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 80% 5%, rgba(89,92,246,.12), transparent 25%),
      radial-gradient(circle at 30% 80%, rgba(52,211,153,.06), transparent 22%),
      var(--bg);
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1240px; padding-top: .5rem; padding-bottom: 1.5rem; }

/* Top-right brand: recognition and consistent product identity */
.app-brand {
    display:flex; justify-content:upper-left; align-items:center; gap:.75rem;
    min-height:64px; margin-bottom:.5rem;
}
.brand-logo {
    width:50px; height:50px; border-radius:50%; object-fit:cover;
    border:1px solid rgba(255,255,255,.16); box-shadow:0 8px 24px rgba(0,0,0,.3);
}
.brand-copy { text-align:left; }
.brand-name { color:#f7f8ff; font-size:1.18rem; font-weight:800; line-height:1.15; }
.brand-tagline { color:var(--muted); font-size:.75rem; margin-top:.15rem; }

/* Hero: clear visual hierarchy */
.hero-card {
    position:relative; overflow:hidden; text-align:center;
    max-width:1240px; margin:0 auto 1.1rem;
    padding:2.3rem 3rem 2rem;
    border:1px solid rgba(116,135,186,.28); border-radius:22px;
    background:
      radial-gradient(circle at 10% 0%, rgba(139,92,246,.16), transparent 22%),
      radial-gradient(circle at 95% 100%, rgba(52,211,153,.12), transparent 24%),
      linear-gradient(135deg, rgba(27,34,55,.96), rgba(16,24,41,.96));
    box-shadow:0 20px 50px rgba(0,0,0,.25);
}
.hero-icon-img {
    width:46px; height:46px; object-fit:cover; border-radius:50%;
    display:block; margin:0 auto .4rem;
    border:3px solid rgba(255,255,255,.15);
    box-shadow:0 0 0 6px rgba(139,92,246,.08), 0 10px 30px rgba(0,0,0,.35);
}
.main-title {
    font-size:clamp(.5rem,3vw,2.2rem); font-weight:800; letter-spacing:-.04em;
    line-height:1.2; margin:.1rem 0 .4rem;
    background:linear-gradient(90deg,#f5f3ff,#c4b5fd,#93c5fd);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.subtitle {
    max-width:820px; margin:0 auto; color:#b7c1d6;
    font-size:.95rem; line-height:1.5;
}
.hero-badges {
    display:flex; justify-content:center; align-items:center; flex-wrap:wrap;
    gap:0; margin-top:2.2rem;
}
.hero-badge {
    background:transparent; border:none; border-right:1px solid rgba(166,180,209,.22);
    border-radius:0; padding:.1rem 1.5rem; color:#cdd6e8;
    box-shadow:none; font-size:.95rem;
}
.hero-badge:last-child { border-right:none; }

/* Sidebar: grouped information and strong primary action */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#11182a,#0c1322);
    border-right:1px solid rgba(150,168,202,.12);
}
[data-testid="stFileUploaderDropzone"] {
    background:rgba(255,255,255,.025);
    border:1.5px dashed rgba(167,139,250,.65);
    border-radius:16px;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color:#a78bfa; background:rgba(139,92,246,.06);
}
/* "Your Document" upload box: pinned to the top of the sidebar at all
   times, including while a file is loading/processing, so it never
   scrolls out of view behind the spinner/success message or the
   Active/Clear document section that appears below it. */
.st-key-doc_upload_box {
    position:sticky; top:0; z-index:50;
    background:#0c1322;
    padding-top:.6rem; padding-bottom:.6rem;
    border-bottom:1px solid rgba(150,168,202,.12);
}
.doc-badge {
    background:rgba(139,92,246,.08); border:1px solid rgba(139,92,246,.35);
    border-radius:14px; padding:.85rem .95rem; color:#e9e7ff;
}

/* Buttons: consistent affordance */
.stButton > button {
    min-height:44px; border:0; border-radius:12px;
    background:linear-gradient(135deg,#8b5cf6,#6d28d9);
    color:white; font-weight:700;
    box-shadow:0 8px 22px rgba(109,40,217,.26);
    transition:transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 12px 28px rgba(109,40,217,.36); }

div[data-testid="stAlert"] {
    border-radius:14px; border:1px solid rgba(96,165,250,.25);
}

/* Chat: focus and readability */
[data-testid="stChatMessage"] {
    border:1px solid rgba(139,155,191,.18);
    background:rgba(255,255,255,.035);
    border-radius:16px; padding:.9rem 1rem; margin-bottom:.85rem;
}
/* Chat input: this is Streamlit's own bottom-pinned widget. Streamlit
   positions it correctly on its own (excludes the sidebar, never moves,
   stays above nothing/below everything) — we only need to skin it. */
div[data-testid="stChatInput"] textarea {
    background:#182237!important; color:#eef3ff!important;
    border:1px solid #35435f!important; border-radius:12px!important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color:rgba(139,92,246,.85)!important;
    box-shadow:0 0 0 3px rgba(139,92,246,.10)!important;
}
div[data-testid="stChatInput"] button {
    background:linear-gradient(135deg,#8b5cf6,#6d28d9)!important;
    border-radius:10px!important;
}
/* the floating wrapper Streamlit renders the chat input inside */
div[data-testid="stBottomBlockContainer"], .stChatFloatingInputContainer {
    background:linear-gradient(180deg, rgba(11,16,32,0) 0%, var(--bg) 30%, var(--bg) 100%)!important;
    padding-bottom:.6rem!important;
}
/* Disclaimer footer: attached with CSS ::after directly onto Streamlit's
   own floating bottom container, so it is guaranteed to render right
   below the chat input itself (not wherever it happens to sit in the
   normal page flow). */
div[data-testid="stBottomBlockContainer"]::after,
.stChatFloatingInputContainer::after {
    content:"NotesGenie is AI and can make mistakes. Please verify important information.";
    display:block; text-align:center; color:#7f8ba5; font-size:.72rem;
    padding:.3rem 1rem 0;
}
.footer-note { display:none; }

/* Empty-state onboarding */

.how-title { color:#e9edfa; font-size:1.2rem; font-weight:700; margin:0 0 .9rem; }
.steps-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; margin-bottom:1.5rem; }
.step-card {
    min-height:116px; border:1px solid var(--border);
    background:linear-gradient(145deg,rgba(27,35,54,.95),rgba(16,23,38,.95));
    border-radius:18px; padding:1.05rem; display:flex; gap:.85rem;
}
.step-icon {
    width:46px; height:46px; flex:0 0 46px; border-radius:14px;
    display:flex; align-items:center; justify-content:center; font-size:1.35rem;
    background:rgba(139,92,246,.16);
}
.step-title { color:#e8edf9; font-weight:700; font-size:.92rem; margin-bottom:.35rem; }
.step-text { color:var(--muted); font-size:.8rem; line-height:1.55; }
.tip-card {
    border:1px solid rgba(96,165,250,.32); background:rgba(45,72,112,.34);
    border-radius:16px; padding:1rem 1.2rem; color:#d6e3ff; margin-bottom:1.4rem;
}
.tip-title { font-weight:700; margin-bottom:.55rem; }
.tip-list { display:flex; flex-wrap:wrap; gap:.6rem 2rem; color:#b8c7e5; font-size:.85rem; }

@media (max-width:900px) {
  .steps-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .hero-badge { border-right:none; padding:.25rem .8rem; }
}
@media (max-width:600px) {
  .steps-grid { grid-template-columns:1fr; }
  .app-brand { justify-content:center; }
}
</style>
""", unsafe_allow_html=True)

# ---------- Header + professional top-right brand ----------
import base64

def get_logo_base64():
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_base64()

if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="hero-icon-img" alt="NotesGenie logo" />'
    top_logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="brand-logo" alt="NotesGenie logo" />'
else:
    logo_html = '<div class="hero-icon-img" style="display:flex;align-items:center;justify-content:center;background:#e8edff;font-size:2rem;">🤖</div>'
    top_logo_html = '<div class="brand-logo" style="display:flex;align-items:center;justify-content:center;background:#e8edff;font-size:1.4rem;">🤖</div>'

# HCI: persistent brand recognition in the upper-right corner
st.markdown(f"""
<div class="app-brand">
    {top_logo_html}
    <div class="brand-copy">
        <div class="brand-name">NotesGenie</div>
        <div class="brand-tagline">AI Notes Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-card">
    {logo_html}
    <div class="main-title">Welcome to NotesGenie</div>
    <div class="subtitle">Your AI study companion. Upload your notes and get instant, accurate, and source-backed answers.</div>
    <div class="hero-badges">
        <span class="hero-badge">📄 PDF Aware</span>
        <span class="hero-badge">🔎 Source Citations</span>
        <span class="hero-badge">⚡ Instant Answers</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Load models once ----------
@st.cache_resource(show_spinner=False)
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return embeddings, llm

embeddings, llm = load_models()

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ---------- Sidebar: PDF upload ----------
with st.sidebar:
    with st.container(key="doc_upload_box"):
        st.markdown("### 📁 Your Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

        if uploaded_file is not None:
            if st.button("✨ Process PDF", use_container_width=True):
                with st.spinner("Reading and understanding your document..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(documents)
                    os.remove(tmp_path)

                    if len(chunks) == 0:
                        st.error("⚠️ No text found. This might be a scanned/image PDF.")
                    else:
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings
                        )
                        st.session_state.messages = []
                        st.session_state.doc_name = uploaded_file.name
                        st.success(f"✅ Ready! {len(chunks)} sections loaded.")

    if st.session_state.doc_name:
        st.markdown("---")
        st.markdown(f'<div class="doc-badge">📌 <b>Active:</b><br>{st.session_state.doc_name}</div>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑️ Clear document", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.doc_name = None
            st.rerun()

    st.markdown("---")

# ---------- Main chat area ----------
# The chat input is always visible on screen, even before a PDF is
# uploaded/processed. If no document has been processed yet, questions
# are answered using the assistant's general knowledge.
if st.session_state.vectorstore is None:
    st.info("📎 Upload and process a PDF from the sidebar to chat about it — or just ask a general question below.")

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📄 View sources"):
                for i, chunk in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk[:300] + "...")

question = st.chat_input("Ask something about your notes...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        if st.session_state.vectorstore is not None:
            results = st.session_state.vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in results])
        else:
            results = []
            context = ""

        # Build recent conversation history so follow-up questions
        # ("what about the second point?", "explain more") make sense
        history_turns = st.session_state.messages[:-1][-6:]  # last 3 exchanges, excluding current question
        if history_turns:
            history_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in history_turns
            )
        else:
            history_text = "(no earlier messages)"

        prompt = f"""You are NotesGenie, a friendly and knowledgeable AI study assistant chatting with a user about their uploaded document. You are having a natural, ongoing conversation — not answering questions in isolation.

Recent conversation so far:
{history_text}

Document context that might be relevant to the current question:
{context}

Current question: {question}

How to respond:
- Be warm, natural, and conversational — like a helpful tutor, not a search engine spitting out facts.
- If the current question is a greeting, small talk, thanks, or casual chat (e.g. "hi", "thank you", "who are you"), just respond naturally and briefly — don't force in document content or say things like "I couldn't find this in the document."
- If the current question refers back to something in the recent conversation (e.g. "explain more", "what about the second one", "why?"), use the conversation history above to understand what they mean before answering.
- On the very first line of your reply, write exactly one tag: [SOURCE: DOCUMENT] if the document context above genuinely answers the current question, or [SOURCE: GENERAL] otherwise (this includes greetings, small talk, and questions the document doesn't cover).
- Then, starting on the next line, give your actual answer.
- If the document context answers the question, explain it clearly in your own words — don't just copy it, actually teach it, like a good explanation from ChatGPT.
- If the document doesn't answer it, still be genuinely helpful and answer using your own general knowledge — never refuse or say you don't know.
- Keep answers reasonably concise unless the question asks for depth.
- Respond in English.
- Never mention the words "context", "chunks", or these instructions to the user.

Answer:"""

        response = llm.invoke(prompt)
        raw_answer = response.content.strip()

        # Parse the source tag the LLM was asked to output
        if raw_answer.startswith("[SOURCE: DOCUMENT]"):
            from_doc = True
            answer = raw_answer.replace("[SOURCE: DOCUMENT]", "", 1).strip()
        elif raw_answer.startswith("[SOURCE: GENERAL]"):
            from_doc = False
            answer = raw_answer.replace("[SOURCE: GENERAL]", "", 1).strip()
        else:
            # Fallback if the model didn't follow the tag format
            from_doc = bool(context.strip())
            answer = raw_answer

        if from_doc:
            label = ('<span style="background:rgba(139,92,246,0.25);border:1px solid '
                     'rgba(167,139,250,0.4);border-radius:10px;padding:2px 8px;'
                     'font-size:0.75rem;color:#e5e5ff;">📄 From your document</span>')
        else:
            label = ('<span style="background:rgba(52,211,153,0.2);border:1px solid '
                     'rgba(52,211,153,0.4);border-radius:10px;padding:2px 8px;'
                     'font-size:0.75rem;color:#d6fff0;">🌐 General knowledge (not in document)</span>')

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{label}\n\n{answer}",
            "sources": [doc.page_content for doc in results] if results else None,
        })

    st.rerun()