import streamlit as st
import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import re
import base64
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI


# ----------------------------
# Configuration
# ----------------------------
DATA_FOLDER = "data"
VECTOR_DB_PATH = "vectorstore"

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------------
# Load or Ingest Vector DB
# ----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.6")

@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

@st.cache_resource
def load_or_create_vectorstore():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, load_embeddings(),allow_dangerous_deserialization=True)
    
    docs = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_FOLDER, filename))
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["filename"] = filename
            docs.extend(loaded_docs)
    
    splitter = get_text_splitter()
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, load_embeddings())
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore

# ----------------------------
# Semantic Search
# ----------------------------
def search_resumes(query, score_threshold=0.5):
    vs = load_or_create_vectorstore()
    results = vs.similarity_search_with_score(query, k=10)
    
    grouped = []
    seen = set()

    for doc, score in results:
        if score < score_threshold:
            continue
        filename = doc.metadata.get("filename", "unknown")
        if filename in seen:
            continue
        seen.add(filename)
        grouped.append({
            "filename": filename,
            "text": doc.page_content,
            "score": score
        })
    return pd.DataFrame(grouped)



# ----------------------------
# Gemini QA with RAG
# ----------------------------
def ask_question_rag(question):
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured."

    vs = load_or_create_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI assistant reviewing job candidates.
Based on the following extracted resume content:

{context}

Answer this question:
{question}
"""
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain.run(question)

# ----------------------------
# Streamlit UI Logic (Keep Frontend Same)
# ----------------------------
st.set_page_config(page_title="AI Resume Search", layout="wide")
st.title("🔍 AI-Powered Resume Search Engine")

query = st.text_input("💼 Enter Job Role or Description:")

if query:
    with st.spinner("📚 Processing resumes..."):
        results = search_resumes(query)

        if not results.empty:
            skills_list, projects_list, education_list, exp_list = [], [], [], []

            for _, row in results.iterrows():
                skills, projects, edu, exp = extract_keywords(row["text"])
                skills_list.append(skills)
                projects_list.append(projects)
                education_list.append(edu)
                exp_list.append(exp)

            results["Skills"] = skills_list
            results["Projects"] = projects_list
            results["Education"] = education_list
            results["Experience"] = exp_list

            edu_filter = st.multiselect("🎓 Filter by Education:", sorted(set(education_list)))
            exp_filter = st.multiselect("💼 Filter by Experience:", sorted(set(exp_list)))

            if edu_filter:
                results = results[results["Education"].isin(edu_filter)]
            if exp_filter:
                results = results[results["Experience"].isin(exp_filter)]

            st.success(f"✅ Found {len(results)} matching resume(s).")
            st.dataframe(results[["filename", "Skills", "Projects", "Education", "Experience", "score"]])

            for _, row in results.iterrows():
                resume_path = os.path.join(DATA_FOLDER, row["filename"])
                base64_pdf = get_base64_pdf(resume_path)
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

                with st.expander(f"📄 View {row['filename']}"):
                    st.markdown(pdf_display, unsafe_allow_html=True)
                    st.download_button("⬇️ Download Resume", data=base64.b64decode(base64_pdf),
                                       file_name=row["filename"], mime="application/pdf")

            # RAG QA Section
            st.subheader("🧠 Ask a question over these resumes")
            user_question = st.text_input("Type your question:")

            if user_question:
                with st.spinner("💬 Generating answer..."):
                    answer = ask_question_rag(user_question)
                    st.success("💬 Gemini RAG Answer:")
                    st.markdown(f"> {user_question}\n\n{answer}")
        else:
            st.warning("No matching resumes found.")