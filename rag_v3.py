import os

from dotenv import load_dotenv
from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# =========================
# Environment
# =========================

load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "your_project_name"

PDF_PATH = r"your_pdf_path"

DATABASE_URL = os.getenv("DATABASE_URL")

COLLECTION_NAME = "pdf_name"


# =========================
# Embeddings
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# Vector Store
# =========================

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    use_jsonb=True
)


# =========================
# Check if PDF is already stored
# =========================

@traceable(name="setup_vectorstore")
def setup_vectorstore():

    # Check existing documents
    existing_docs = vectorstore.similarity_search(
        "document",
        k=1
    )

    if existing_docs:
        print("PDF already exists in PostgreSQL.")
        print("Skipping PDF loading and embedding.")

        return vectorstore

    # First time only
    print("First time setup...")
    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("Splitting PDF...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    splits = splitter.split_documents(docs)

    print("Total chunks:", len(splits))

    print("Creating embeddings and storing in PostgreSQL...")

    vectorstore.add_documents(splits)

    print("PDF stored successfully.")

    return vectorstore


# =========================
# Setup
# =========================

vectorstore = setup_vectorstore()


# =========================
# Retriever
# =========================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================
# Prompt
# =========================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer only from the provided context.

        If the answer is not present in the context,
        say you don't know.
        """
    ),
    (
        "human",
        """
        Question: {question}

        Context:
        {context}
        """
    )
])


# =========================
# Format documents
# =========================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# =========================
# RAG Chain
# =========================

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})


chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)


# =========================
# Ask Question
# =========================

print("\nPDF RAG ready.")

while True:

    question = input("\nQ: ")

    if question.lower() == "exit":
        break

    answer = chain.invoke(
        question,
        config={
            "run_name": "pdf_rag_query"
        }
    )

    print("\nA:", answer)