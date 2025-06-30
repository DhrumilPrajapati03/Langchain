from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load the Hugging Face token from .env file
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # supports chat completions
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the occupation, age and city of eva elfie \n {format_instruction}",
    input_variables=["format_instruction"],
    partial_variables={"format_instruction": parser.get_format_instructions() + "\n\n"}
)

prompt = template.format()

print(prompt)

result = model.invoke(prompt)

print(result)