import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PyPDF2 import PdfReader

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=api_key, temperature=0)

# Load and extract resume text
def load_resume(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format")

resume_text = load_resume("Sample_resume.txt") 

# Ask questions
while True:
    query = input("\nAsk a question about your resume (or type 'exit'): ")
    if query.lower() in ['exit', 'quit']:
        break

    prompt = PromptTemplate(
        input_variables=["resume", "question"],
        template="""
You are an AI assistant reading the following resume:

{resume}

Answer the question below as accurately as possible:

Question: {question}
"""
    )

    final_prompt = prompt.format(resume=resume_text, question=query)
    message = HumanMessage(content=final_prompt)
    response = llm.invoke([message])
    print(f"\nAnswer: {response.content}")
