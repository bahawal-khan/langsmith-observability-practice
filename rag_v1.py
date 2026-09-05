
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()


os.environ["LANGSMITH_PROJECT"] = "your_project_name"

print("API Key:", bool(os.getenv("LANGSMITH_API_KEY")))


# PDF path
PDF_PATH = r"your_pdf_path"

# --------------------------------------------------
# 1) Load PDF
# --------------------------------------------------

loader = PyPDFLoader(PDF_PATH)

docs = loader.load()

print(f"Pages loaded: {len(docs)}")


# --------------------------------------------------
# 2) Split documents into chunks
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

splits = splitter.split_documents(docs)

print(f"Chunks created: {len(splits)}")


# --------------------------------------------------
# 3) Create embeddings + FAISS vector store
# --------------------------------------------------

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vs = FAISS.from_documents(
    splits,
    emb
)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)


# --------------------------------------------------
# 4) Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If the answer is not found in the context, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\n"
        "Context:\n{context}"
    )
])


# --------------------------------------------------
# 5) Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# 6) Format retrieved documents
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# 7) Parallel retrieval
# --------------------------------------------------

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})


# --------------------------------------------------
# 8) RAG Chain
# --------------------------------------------------

chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)


# --------------------------------------------------
# 9) Ask questions
# --------------------------------------------------

print("\nPDF RAG ready.")
print("Type your question or 'exit' to quit.")


while True:

    q = input("\nQ: ").strip()

    if q.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    if not q:
        continue

    ans = chain.invoke(q)

    print("\nA:", ans)