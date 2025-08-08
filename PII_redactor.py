from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

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

# Define structured output format with Pydantic
class EntityRecognitionResponse(BaseModel):
    phone_number: str = Field(..., description="Phone number extracted from the input text.")
    credit_card_number: str = Field(..., description="Credit card number extracted from the input text.")
    social_security_number: str = Field(..., description="Social Security Number extracted from the input text.")
    name: str = Field(..., description="Name extracted from the input text.")
    address: str = Field(..., description="Address extracted from the input text.")
    masked_text: str = Field(..., description="Masked version of the input text with all sensitive data redacted.")

# Main function
def main():
    llm = initialize_llm()
    llm_with_structured_output = llm.with_structured_output(EntityRecognitionResponse)

    # Updated prompt with SSN and redaction instructions
    prompt_template = """
    You are an advanced Named Entity Recognition (NER) system that extracts and redacts sensitive information.

    Your task is to:
    1. Extract the following entities from the provided text:
    - Full Name
    - Address
    - Phone Number
    - Credit Card Number
    - Social Security Number

    2. Redact **only** the following entities in the text:
    - Address
    - Credit Card Number
    - Social Security Number

    In the redacted version of the text, replace each of the above sensitive fields with “[REDACTED]”, even if they appear multiple times.

    Return the following JSON object:
    {{
        "name": "<extracted name>",
        "address": "<extracted address>",
        "phone_number": "<extracted phone>",
        "credit_card_number": "<extracted credit card>",
        "social_security_number": "<extracted SSN>",
        "masked_text": "<full original text with only the address, SSN, and credit card number replaced by [REDACTED]>"
    }}

    ONLY return the JSON object, and nothing else.

    Text:
    {text}
    """

    prompt = PromptTemplate.from_template(prompt_template)
    entity_chain = prompt | llm_with_structured_output

    # Sample input with PII
    text = """John Smith lives at 742 Evergreen Terrace, Springfield, IL 62704. 
    You can reach him at (217) 555-0198 or via email at john.smith@email.com. 
    He works at Globex Corporation as a Software Engineer.  
    His credit card number is 3782 822463 10005 and it expires on 09/26.  
    John's social security number is 123-45-6789.
    """

    # Get the response
    response = entity_chain.invoke({"text": text})

    print("\n📦 Extracted Entities:")
    print(f"Name: {response.name}")
    print(f"Address: {response.address}")
    print(f"Phone Number: {response.phone_number}")
    print(f"Credit Card Number: {response.credit_card_number}")
    print(f"Social Security Number: {response.social_security_number}")
    print(f"Masked Text: {response.masked_text}")
if __name__ == "__main__":
    main()
