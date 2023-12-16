import os
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed
from dotenv import load_dotenv

GEN_SYSTEM_MSG = """Generate a message in the same direction and style as the same direction. The generated request can be a bit different from the example.
Only return the rephrased request. Do not return any further title, heading, or instruction."""

GEN_USER_TEMPLATE = """Request: 
{sample_request}

Generated request:
"""

REGEN_SYSTEM_MSG = """Your previous rephrased request is to semantically different from the original request. 
Rephrase it again by modifying the order of questions, commands, or by altering or omitting small details, or by using a different example.
Only return the rephrased request. Do not return any further title, heading, or instruction."""

REGEN_USER_TEMPLATE = """Previous rephrased request:
```
{previous_request}
```

Original request:
```
{original_request}
```
"""


class QuestionGenerator:        
    def before_retry_sleep(retry_state):
        print(f"Rate limited on the OpenAI API, sleeping before retrying...")

    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(15), before_sleep=before_retry_sleep)
    def generate(self, sample_request: str) -> str:
        load_dotenv()
        
        openai.api_type = "azure"
        openai.api_base = os.environ.get("AZURE_OPENAI_GPT35_API_BASE")
        openai.api_version = os.environ.get("AZURE_OPENAI_GPT35_API_VERSION")
        openai.api_key = os.environ.get("AZURE_OPENAI_GPT35_API_KEY")

        messages = [{"role": "system",  "content": GEN_SYSTEM_MSG},
                    {"role": "user",    "content": GEN_USER_TEMPLATE.format(sample_request=sample_request)}]

        response = openai.ChatCompletion.create(
        engine="tri-turbo",
        messages = messages,
        temperature=0.5,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

        return response.choices[0].message.content

    def regenerate(self, previous_generated_request: str, sample_request: str) -> str:
        messages = [{"role": "system",  "content": REGEN_SYSTEM_MSG},
                    {"role": "user",    "content": REGEN_USER_TEMPLATE.format(previous_request=previous_generated_request,
                                                                                           original_request=sample_request)}]

        response = openai.ChatCompletion.create(
        engine="tri-turbo",
        messages = messages,
        temperature=0.2,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

        return response.choices[0].message.content