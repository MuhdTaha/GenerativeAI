from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

# Initialize the language model
def initialize_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        max_completion_tokens=4000,
        temperature=0.1,
    )

# Initialize your LLM
llm = initialize_llm()

# Translation prompt template
prompt_template = PromptTemplate(
    input_variables=["text", "target_language"],
    template=(
        "Translate the following text from English to {target_language}:\n\n"
        "{text}\n\n"
        "Only provide the translated text without extra explanations."
    )
)

# Ask the user for inputs
english_text = input("\nEnter the text in English: ").strip()
target_language = input("\nEnter the language you want to translate into: ").strip()

# Build the final prompt
prompt = prompt_template.format(text=english_text, target_language=target_language)

# Run the LLM
response = llm.invoke(prompt)

# Display the result
print("\n\n📄 Translated Text:")
print(response.content)
print()