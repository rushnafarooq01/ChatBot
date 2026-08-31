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
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS: 3D / glassmorphism look ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    html, body {
        background: #16141f !important;
    }

    /* Flat, solid background — no animated gradient patches */
    .stApp {
        background: #16141f !important;
    }

    /* Hide Streamlit's default header/menu/footer for a clean custom look */
    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
    }
    footer {visibility: hidden;}

    /* 3D floating title card */
    .hero-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        padding: 2rem 1.5rem 1.6rem 1.5rem;
        margin-top: 0.5rem;
        margin-bottom: 1.8rem;
        text-align: center;
        box-shadow:
            0 20px 40px rgba(0, 0, 0, 0.45),
            0 2px 0 rgba(255, 255, 255, 0.08) inset,
            0 -2px 20px rgba(124, 58, 237, 0.15) inset;
        position: relative;
        overflow: hidden;
    }

    /* Glow orb decorations inside hero card */
    .hero-card::before {
        content: "";
        position: absolute;
        top: -60px;
        left: -40px;
        width: 160px;
        height: 160px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.35), transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-card::after {
        content: "";
        position: absolute;
        bottom: -70px;
        right: -50px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.25), transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-icon-img {
        width: 76px;
        height: 76px;
        border-radius: 50%;
        object-fit: cover;
        display: block;
        margin: 0 auto 0.4rem auto;
        filter: drop-shadow(0 6px 14px rgba(167, 139, 250, 0.55));
        border: 2px solid rgba(255, 255, 255, 0.15);
    }

    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 0.9rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.75rem;
        color: #d8d8f5;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.3;
        background: linear-gradient(90deg, #c4b5fd, #93c5fd, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        padding-bottom: 0.2rem;
    }

    .subtitle {
        color: #c7c7e0;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Theme Streamlit's fixed bottom chat input bar to match dark design */
    [data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer,
    [data-testid="stBottom"] {
        background: #16141f !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        padding-bottom: 0.5rem !important;
        padding-top: 0.6rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: none;
    }

    /* Remove any extra wrapper borders/backgrounds around the chat input
       so only the textarea itself is visible, no boxed-in-a-box look */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Disclaimer line shown right below the chat input */
    [data-testid="stBottomBlockContainer"]::after {
        content: "NotesGenie is AI and can make mistakes.";
        display: block;
        text-align: center;
        width: 100%;
        color: #8b8bb0;
        font-size: 0.72rem;
        margin-top: 0.5rem;
    }

    /* Center and constrain the chat input width, float it above the edge */
    [data-testid="stChatInput"] {
        max-width: 680px;
        width: 90%;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] > div {
        background: transparent !important;
        width: 100%;
    }
    [data-testid="stChatInput"] textarea {
        background: #26243a !important;
        color: #f0f0f5 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border: 1px solid rgba(167, 139, 250, 0.5) !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8a8aa0 !important;
    }
    [data-testid="stChatInput"] button {
        background: #34324a !important;
        border-radius: 10px !important;
        border: none !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: #40405c !important;
    }

    /* Extra bottom space so chat messages never hide behind the input bar */
    .block-container {
        padding-bottom: 7.5rem;
    }

    /* Chat bubbles - raised 3D card effect */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow:
            0 10px 20px rgba(0, 0, 0, 0.35),
            0 1px 0 rgba(255, 255, 255, 0.08) inset;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow:
            0 14px 26px rgba(0, 0, 0, 0.45),
            0 1px 0 rgba(255, 255, 255, 0.1) inset;
    }

    /* Sidebar - raised panel look */
    section[data-testid="stSidebar"] {
        background: #1a1826 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: none;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] .stMarkdown {
        color: #d8d8f0;
    }

    /* File uploader - embossed dashed card */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.04);
        border: 2px dashed rgba(167, 139, 250, 0.5);
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3) inset;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(167, 139, 250, 0.9);
    }

    /* Buttons - glossy 3D */
    .stButton > button {
        background: linear-gradient(180deg, #8b5cf6, #6d28d9);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.1rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        box-shadow:
            0 6px 14px rgba(109, 40, 217, 0.5),
            0 1px 0 rgba(255, 255, 255, 0.3) inset,
            0 -3px 8px rgba(0, 0, 0, 0.25) inset;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 10px 22px rgba(109, 40, 217, 0.65),
            0 1px 0 rgba(255, 255, 255, 0.35) inset,
            0 -3px 8px rgba(0, 0, 0, 0.25) inset;
    }
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 3px 8px rgba(109, 40, 217, 0.4) inset;
    }

    /* Chat input - floating pill */
    [data-testid="stChatInput"] {
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
    }

    /* Alerts - depth cards */
    div[data-testid="stAlert"] {
        border-radius: 14px;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Expander - raised panel */
    details {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(167, 139, 250, 0.2);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.3);
    }

    /* Active document badge */
    .doc-badge {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(37, 99, 235, 0.25));
        border: 1px solid rgba(167, 139, 250, 0.4);
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.3);
        color: #e5e5ff;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header (3D hero card) ----------
import base64

def get_logo_base64():
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_base64()

if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="hero-icon-img" />'
else:
    logo_html = '<span class="hero-icon">📘</span>'

st.markdown(f"""
<div class="hero-card">
    {logo_html}
    <div class="main-title">NotesGenie</div>
    <div class="subtitle">Upload your notes. Ask anything. Get instant, source-backed answers.</div>
    <div class="hero-badges">
        <span class="hero-badge">📄 PDF Aware</span>
        <span class="hero-badge">🔍 Source Citations</span>
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
    st.markdown("## 📄 Your Document")
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
if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar and click **Process PDF** to get started.")
    st.chat_input("Upload a PDF first to start chatting...", disabled=True)
else:
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about your notes...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching your notes..."):
                results = st.session_state.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in results])

                prompt = f"""Answer the question based only on the context below.
Respond in English.
If the answer is not found in the context, say "I couldn't find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

                response = llm.invoke(prompt)
                answer = response.content
                st.markdown(answer)

                with st.expander("📄 View sources"):
                    for i, doc in enumerate(results):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})