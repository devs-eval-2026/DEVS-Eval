import replicate
import google.generativeai as genai
import os
import requests
import boto3
import sagemaker
import json
import subprocess
import time
import re
import logging

_logger = logging.getLogger("devs-eval")

# --- Replicate (CodeLlama / Wizard): 429 = rate limit; 6 req/min is common on low-credit plans ---


def _replicate_codellama_input(preprompt: str, prompt: str) -> dict:
    return {
        "top_k": 250,
        "top_p": 0.95,
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.95,
        "system_prompt": preprompt,
        "repeat_penalty": 1.1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }


def _replicate_run_stream(model_ref: str, input_dict: dict, max_attempts: int = 25) -> str:
    """Waits and retries on 429/throttle (up to ~6 req/min on the limited plan)."""
    for attempt in range(max_attempts):
        try:
            out_iter = replicate.run(model_ref, input=input_dict)
            s = ""
            for item in out_iter:
                s += item
            print(s)
            if not s.strip():
                _logger.warning(
                    "Replicate returned empty text (0 tokens) for %s. "
                    "Possible causes: blocked prompt content, silent model failure, or unexpected response.",
                    model_ref,
                )
            return s
        except Exception as e:
            msg = str(e)
            print(msg)
            if "429" in msg or "throttl" in msg.lower() or "rate limit" in msg.lower():
                wait = 12
                m = re.search(r"~?(\d+)\s*s", msg, re.I)
                if m:
                    wait = max(int(m.group(1)) + 2, 10)
                print(f"^ Replicate rate-limited (429). Waiting {wait}s…")
                time.sleep(wait)
                continue
            if "status: 502" in msg or "Prediction interrupted" in msg:
                time.sleep(10)
                continue
            _logger.error(
                "Replicate failed (will not retry on this exception) — %s: %s. Model: %s",
                type(e).__name__,
                msg[:800],
                model_ref,
            )
            return ""
    _logger.error(
        "Replicate: exhausted %d attempts (429/502) for %s; returning empty string.",
        max_attempts,
        model_ref,
    )
    return ""


# GPT evaluation generation
# set your API key here, need to be hided
def GPT5_4(preprompt, prompt, client):
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": prompt},
    ]

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=messages,
            )
            choice = response.choices[0]
            content = choice.message.content
            if content is None:
                fr = getattr(choice, "finish_reason", None)
                _logger.warning(
                    "OpenAI (GPT5_4, gpt-5.4): message.content is null. finish_reason=%r",
                    fr,
                )
                return ""
            s_out = str(content)
            if not s_out.strip():
                _logger.warning(
                    "OpenAI (GPT5_4, gpt-5.4): response returned empty text after success."
                )
            return s_out
        except Exception as e:
            s = str(e)
            if "Rate limit is exceeded" in s:
                time.sleep(30)
                continue
            _logger.error(
                "OpenAI (GPT5_4) failed — %s: %s",
                type(e).__name__,
                s[:800],
            )
            return ""

def GPT5_3_Codex(preprompt, prompt, client):
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": prompt},
    ]
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-5.3-codex",
                messages=messages,
            )
            choice = response.choices[0]
            content = choice.message.content
            if content is None:
                fr = getattr(choice, "finish_reason", None)
                _logger.warning(
                    "OpenAI (GPT5_3_Codex, gpt-5.3-codex): message.content is null. finish_reason=%r",
                    fr,
                )
                return ""
            s_out = str(content)
            if not s_out.strip():
                _logger.warning(
                    "OpenAI (GPT5_3_Codex, gpt-5.3-codex): response returned empty text after success."
                )
            return s_out
        except Exception as e:
            s = str(e)
            if "Rate limit is exceeded" in s:
                time.sleep(30)
                continue
            _logger.error(
                "OpenAI (GPT5_3_Codex) failed — %s: %s",
                type(e).__name__,
                s[:800],
            )
            return ""

def Codellama7b(preprompt, prompt):
    return _replicate_run_stream(
        "meta/codellama-7b-instruct:aac3ab196f8a75729aab9368cd45ea6ad3fc793b6cda93b1ded17299df369332",
        _replicate_codellama_input(preprompt, prompt),
    )



def Codellama34b(preprompt, prompt):
    return _replicate_run_stream(
        "meta/codellama-34b-instruct:eeb928567781f4e90d2aba57a51baef235de53f907c214a4ab42adabf5bb9736",
        _replicate_codellama_input(preprompt, prompt),
    )


# WizardCoder-Python-34B-V1.0	https://replicate.com/lucataco/wizardcoder-python-34b-v1.0

# WizardCoder-33B-V1.1	: https://replicate.com/lucataco/wizardcoder-33b-v1.1-gguf
def Wizardcoder33b(preprompt, prompt):
    return _replicate_run_stream(
        "lucataco/wizardcoder-33b-v1.1-gguf:bbf93cee2c2b446f0ff426ae81a9b61c5ebd8972a21f734fe035513b6fafe615",
        _replicate_codellama_input(preprompt, prompt),
    )


# WizardCoder-15B-V1.0	https://replicate.com/lucataco/wizardcoder-15b-v1.0
# def Wizardcoder33b(preprompt, prompt):
#     output = replicate.run(
#         "lucataco/wizardcoder-33b-v1.1-gguf:bbf93cee2c2b446f0ff426ae81a9b61c5ebd8972a21f734fe035513b6fafe615",
#         input={
#             "top_k": 250,
#             "top_p": 0.95,
#             "prompt": prompt,
#             "max_tokens": 500,
#             "temperature": 0.95,
#             "system_prompt": preprompt,
#             "repeat_penalty": 1.1,
#             "presence_penalty": 0,
#             "frequency_penalty": 0,
#         },
#     )
#     output_string = ""
#     for item in output:
#         output_string += item
#     print(output_string)
#     return output_string


# Gemini: https://ai.google.dev/gemini-api/docs/quickstart https://github.com/alibaba/CloudEval-YAML/blob/main/models/palm.py
def gemini(preprompt, prompt, model_name: str = "gemini-2.5-flash"):
    """`model_name`: Google AI model ID (e.g. ``gemini-2.5-flash``). The old ``gemini-pro`` is deprecated."""
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    # for m in genai.list_models():
    #     if 'generateContent' in m.supported_generation_methods:
    #         print(m.name)

    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    try:
        generation_config = genai.types.GenerationConfig(max_output_tokens=8192)
    except Exception:
        generation_config = None

    model = genai.GenerativeModel(model_name)
    while True:
        try:
            kwargs = {"safety_settings": safety_settings}
            if generation_config is not None:
                kwargs["generation_config"] = generation_config
            response = model.generate_content(
                preprompt + "\n\n" + prompt,
                **kwargs,
            )
            break
        except Exception as e:
            s = str(e)
            if (
                "Resource has been exhausted" in s
                or "429" in s
                or "quota" in s.lower()
            ):
                print(s)
                print("^ Quota/rate limit hit. Waiting 100 s.")
                time.sleep(100)
            else:
                print("Gemini API error:", s)
                return ""
    # print(len(response.candidates))
    # for candidate in response.candidates:
    #     print(len(candidate.content.parts))
        # print ([part.text for part in candidate.content.parts])
    
    # return response.text
    try:
        if not response.candidates:
            print("Gemini: no candidates returned (e.g. safety block).")
            return ""
        parts = response.candidates[0].content.parts
        if not parts:
            return ""
        return parts[0].text
    except (AttributeError, IndexError) as e:
        print("Gemini: error reading response:", e)
        return ""

# gemini("You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed.", "create a AWS codebuild project resource with example iam role, example GITHUB source, and a logs config")

# print(Magicoder_S_CL_7B("You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed. If variables are used, make sure default values are supplied.", "Here is the actual prompt: Create an AWS VPC resource with an Internet Gateway attached to it"))

# print(Wizardcoder34b("You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed. If variables are used, make sure default values are supplied.", "Here is the actual prompt: Create an AWS VPC resource with an Internet Gateway attached to it"))