import os

import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model="gpt-4o-mini")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",
               "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
               "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
               ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 6}
)

retDocuments = retriever.invoke("agent memory")
chunks = page_contents = [doc.page_content for doc in retDocuments]

# print(len(retDocuments))
# print(retDocuments)

parser = JsonOutputParser()

relevancePrompt = PromptTemplate(
    template="If given all of chunks are relevant with User query return relevance: yes otherwise return relevance: no.\n"
             "{format_instructions}\n"
             "chunks: {context}\n"
             "User query: {question}",
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

userQuery = "agent memory"

chain = relevancePrompt | llm | parser
response = chain.invoke({"context": chunks, "question": userQuery})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def askQuestion():
    askingQuestionPrompt = PromptTemplate(
        template="Answer the question based on the following context\n"
                 "Context: {context}"
                 "Question: {question}",
    )

    askingQuestionChain = (
            {"context": RunnablePassthrough() | (lambda _: chunks), "question": RunnablePassthrough()}
            | askingQuestionPrompt
            | llm
            | StrOutputParser()
    )

    return askingQuestionChain.invoke(userQuery)


if response['relevance'] == 'yes':
    questionResponse = askQuestion()

    hallucinationDetectionPrompt = PromptTemplate(
        template="you are an assistant to decide\n"
                 "whether given answer is hallucination of given question\n"
                 "If It is result of hallucination return hallucination: yes otherwise hallucination: no\n"
                 "{format_instructions}\n"
                 "Question: {question}\n"
                 "Answer: {answer}",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    hallucinationDetectionChain = (
            {"question": RunnablePassthrough(), "answer": RunnablePassthrough() | (lambda _: questionResponse)}
            | hallucinationDetectionPrompt
            | llm
            | parser
    )

    hallucinationDetectionRes = hallucinationDetectionChain.invoke(userQuery)
    # print(hallucinationDetectionRes)

    if hallucinationDetectionRes['hallucination']:
        print("최종 답: ", askQuestion())
    else:
        print("최종 답: ", questionResponse)
else:
    print("관련성이 없습니다")
