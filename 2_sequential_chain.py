from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "sequential_chain"

print("API Key:", bool(os.getenv("LANGSMITH_API_KEY")))

prompt1 = PromptTemplate(
    template="Generate a 5 line report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text\n{text}",
    input_variables=["text"]
)

llm1 = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

llm2 = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

parser = StrOutputParser()

chain = prompt1 | llm1 | parser | prompt2 | llm2 | parser

config = {
    'run_name': 'sequential',
    'tags': ['summarizer'],
    'metadata': {'model_temp': 0}
}

result = chain.invoke({
    "topic": "Which jobs are affected due to ai and in 2030 which jobs have the chance to lose priority because of ai"

},config = config)

print(result)