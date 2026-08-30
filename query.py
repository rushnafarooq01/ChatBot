import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

print("Vector database load ho raha hai...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

print("✅ Ready! Ab sawaal pucho (band karne ke liye 'exit' likho)\n")

while True:
    question = input("Aapka sawaal: ")
    if question.lower() == "exit":
        break

    # Step 1: Sabse relevant chunks dhundo document se
    results = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Step 2: LLM ko context + sawaal bhejo
    prompt = f"""Neeche diye gaye context ke basis par sawaal ka jawab do. 
Agar context mein jawab na mile, to bolo "Mujhe ye jaankari document mein nahi mili."

Context:
{context}

Sawaal: {question}

Jawab:"""

    response = llm.invoke(prompt)
    print("\nJawab:", response.content, "\n")