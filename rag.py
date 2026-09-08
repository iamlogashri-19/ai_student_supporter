import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. PDF Loading & Vector Database
print("1. Loading PDFs and setting up Vector Store...")
loader = PyPDFDirectoryLoader("data/")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

# 2. LLM Setup
print("2. Initializing Llama 3.2...")
llm = ChatOllama(model="llama3.2")


# 3. Fee Calculator Tool Function
def fee_calculator(text):
    numbers = [
        float(n)
        for n in re.findall(r'\b\d+(?:\.\d+)?\b', text)
    ]

    if len(numbers) >= 2:
        tuition = numbers[0]
        hostel = numbers[1]
        years = numbers[2] if len(numbers) >= 3 else 1

        total = (tuition + hostel) * years

        return (
            f"Calculated Total Fee for {int(years)} year(s): "
            f"₹{total:,.2f} "
            f"(Tuition: ₹{tuition:,.2f}, "
            f"Hostel: ₹{hostel:,.2f})"
        )

    return (
        "Please specify tuition fee and hostel fee numbers clearly "
        "(e.g., 'calculate fee for 50000 tuition 30000 hostel 4 years')."
    )


# 4. CGPA Calculator Tool Function
def cgpa_calculator(text):
    numbers = [
        float(n)
        for n in re.findall(r'\b\d+(?:\.\d+)?\b', text)
    ]

    # Numbers should be in pairs:
    # credit, grade point
    if len(numbers) >= 2:

        total_credits = 0
        total_points = 0

        for i in range(0, len(numbers) - 1, 2):
            credit = numbers[i]
            grade_point = numbers[i + 1]

            total_credits += credit
            total_points += credit * grade_point

        if total_credits == 0:
            return "Total credits cannot be zero."

        cgpa = total_points / total_credits

        return f"Your calculated CGPA is: {cgpa:.2f}"

    return (
        "Please provide credit and grade point values. "
        "Example: 'Calculate CGPA: 4 credits 9 grade point, "
        "3 credits 8 grade point'."
    )


# 5. RAG Chain with Memory
system_prompt = (
    "You are an AI Assistant for PET Engineering College. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# 6. Interactive Assistant Loop
print(
    "\n--- Day 3: PETEC Smart Assistant Ready! "
    "(Type 'exit' to quit) ---"
)

session_id = "user_chat_1"

while True:

    user_question = input("\nStudent Question: ")

    if user_question.lower() == 'exit':
        print("Goodbye!")
        break

    if user_question.strip() == "":
        continue

    # Tool Routing Logic

    # CGPA Calculator
    if (
        "cgpa" in user_question.lower()
        or "gpa" in user_question.lower()
    ):
        print("\n[Tool Triggered: CGPA Calculator]")

        ans = cgpa_calculator(user_question)

        print("Assistant Answer:", ans)

    # Fee Calculator
    elif (
        "calculate" in user_question.lower()
        or "total fee" in user_question.lower()
    ):
        print("\n[Tool Triggered: Fee Calculator]")

        ans = fee_calculator(user_question)

        print("Assistant Answer:", ans)

    # RAG Assistant
    else:
        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        print("\nAssistant Answer:", response["answer"])