import os
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed


class QuestionGenerator:
    
    openai.api_type = "azure"
    openai.api_base = "https://tri-testing.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = "302e2d25248b442faa27c9b213f3fb2d"
    
#     system_message = """Rephrase the request by modifying the order of questions, commands, or by altering or omitting small details, or by using a different example.
# Only return the rephrased request. Do not return any further title, heading, or instruction."""

    system_message = """Generate a message in the same direction and style as the same direction. The generated request can be a bit different from the example.
Only return the rephrased request. Do not return any further title, heading, or instruction."""

    user_content_template = """Request: 
{sample_request}

Generated request:
"""

    regen_system_message = """Your previous rephrased request is to semantically different from the original request. 
Rephrase it again by modifying the order of questions, commands, or by altering or omitting small details, or by using a different example.
Only return the rephrased request. Do not return any further title, heading, or instruction."""

    regen_user_content_template = """Previous rephrased request:
```
{previous_request}
```

Original request:
```
{original_request}
```
"""

    def before_retry_sleep(retry_state):
        print(f"Rate limited on the OpenAI API, sleeping before retrying...")

    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(15), before_sleep=before_retry_sleep)
    def generate(self, sample_request: str) -> str:

        messages = [{"role": "system",  "content": self.system_message},
                    {"role": "user",    "content": self.user_content_template.format(sample_request=sample_request)}]

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
        messages = [{"role": "system",  "content": self.regen_system_message},
                    {"role": "user",    "content": self.regen_user_content_template.format(previous_request=previous_generated_request,
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
        

if __name__ == "__main__":
    sample_request = """Hi Dennis, Ali,

Three questions:

1.	We are trying to record/log the choices made on the Educator tablet. [ex: whether the option yes or no was pressed by the researcher]. Is there a way to do this? How?
2.	Somewhat in line with the first question, is there a way to log what the robot is doing moment-to-moment?
3.	Finally, when entering names in the Educator tablet that are accessed by the use of {learner.name} in the Blockly script, is there a setting we can change to enter non-English names? Specifically, we are running a study with Japanese subjects, and can't figure out how to enter names using Japanese characters or with Japanese pronunciation, so names will be pronounced correctly. 

Best,


Name
Job title
Affiliation
Affiliation
email"""

    questionGen = QuestionGenerator()
    genQuestion = questionGen.generate(sample_request)

    print(genQuestion)