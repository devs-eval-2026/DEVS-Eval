from pathlib import Path


_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"


def CoT_prompt(question_prompt):
    with open(_TEMPLATE_DIR / "CoT.txt", "r", encoding="utf-8") as file:
        data = file.read()
    prompt = data + question_prompt  # https://www.promptingguide.ai/techniques/cot#zero-shot-cot-prompting
    return prompt


def FSP_prompt(question_prompt):
    with open(_TEMPLATE_DIR / "few-shot.txt", "r", encoding="utf-8") as file:
        data = file.read()
    prompt = data + question_prompt
    return prompt

def RAG_prompt(context, question_prompt): 
    template = """
Here is some additional knowledge/context retrieved from DEVS documentation, that may (or may not) potentially help you answer the question:
{}

Here is the actual prompt to answer:
{}
    """.format(context, question_prompt)
    return template

def multi_turn_system_prompt():
    with open(_TEMPLATE_DIR / "multi-turn-system-prompt.txt", "r", encoding="utf-8") as file:
        data = file.read()
    return data

def multi_turn_plan_error_prompt(question_prompt, candidate_config, error_message):
    prompt = """
Here is the original prompt:
{}

Here is the incorrect configuration:
{}

Here is the DEVS plan error message (potentially empty):
{}
""".format(question_prompt, candidate_config, error_message)
    return prompt