from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the Hugging Face token from .env file
load_dotenv()

# # Set up the LLM endpoint from Hugging Face
# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # You can replace with any chat-compatible model
#     task="text-generation"  # Must be text-generation for chat models
# )

# # Wrap in LangChain's chat model interface
# model = ChatHuggingFace(llm=llm)

# # Send a prompt
# result = model.invoke("What is the capital of France?")
# print(result.content)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # supports chat completions
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a concise summary on the following topic: {topic}",
    input_variables=["topic"]
)



prompt1 = template1.invoke({"topic": "Climate Change"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"topic": result.content})
result = model.invoke(prompt2)
print(result.content)