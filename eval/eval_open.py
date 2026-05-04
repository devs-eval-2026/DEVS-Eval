import os
import shutil
import csv
import sys
from pathlib import Path
from openai import OpenAI
import numpy as np
import pandas as pd
import subprocess
import json
import ast
import getpass
import uuid
import re
import time
import logging
import models
import prompt_templates
import dataset
from validation import eval_pipeline, empty_code_error, SEMANTIC_JUDGE_SYSTEM
from code_extraction import (
    separate_answer_and_code,
    DEFAULT_DELIMITERS,
    empty_response_log_note,
)
import math

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "retriever"))
)
import llama_index_retriever
import click
from typing import Dict, List, Optional, Tuple

DEFAULT_LOG_FILE = "logs/eval.log"
DELIMITERS = DEFAULT_DELIMITERS

# Default configurations if not specified
NUM_SAMPLES_PER_TASK = 20  # n
EVAL_MODELS = [
    "gpt-5.4",
    "gpt-5.3-codex",
    "gemini-2.5-flash",
    "codellama-7b",
    "codellama-34b",
    "Wizardcoder33b",
]  # default to all available models

# Models that call replicate.run (require REPLICATE_API_TOKEN; do not confuse with OpenAI/Gemini)
MODELS_USING_REPLICATE = frozenset(
    {
        "codellama-7b",
        "codellama-34b",
        "Wizardcoder33b",
    }
)

PROMPT_ENHANCEMENT_STRATS = ["RAG", "COT", "FSP", "multi-turn", ""]

# Evaluation columns per sample (index i in "LLM * #i")
_EVAL_SAMPLE_COL_BASES = (
    "LLM Output #",
    "LLM User Prompt #",
    "LLM System Prompt #",
    "LLM Plannable? #",
    "LLM Correct? #",
    "LLM Notes #",
)


def _replicate_opensource_std_input(preprompt: str, prompt: str) -> dict:
    """Common input for most Code/Wizard models on Replicate (cog-llama)."""
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


def get_quantized_replicate_model_id(eval_model: str) -> Optional[str]:
    """
    Returns the `owner/model:version` identifier on Replicate for the
    **quantized (GGUF/GPTQ/ggml)** variant of the logical model used in eval.

    Can be overridden via environment variables (value: `owner/model:version_hash`):
    EVAL_OPEN_QUANT_CODELLAMA_7B, EVAL_OPEN_QUANT_CODELLAMA_13B,
    EVAL_OPEN_QUANT_CODELLAMA_34B, EVAL_OPEN_QUANT_WIZARD_33B, EVAL_OPEN_QUANT_WIZARD_34B

    7B and 13B have no safe default (models/versions change frequently). Set the env var
    to an active GGUF Cog, or it falls back to the original (non-quantized) ``models``
    in ``replicate_run_opensource_quantized``.
    """
    # Only patterns with proven reusable GGUF/quant defaults; rest configured via env.
    defaults: Dict[str, str] = {
        "codellama-34b": "andreasjansson/codellama-34b-instruct-gguf:1ed692e424acdf3301719be51404611c70e4b99a43b9b0e65b56460490684cac",
        "Wizardcoder33b": "lucataco/wizardcoder-33b-v1.1-gguf:bbf93cee2c2b446f0ff426ae81a9b61c5ebd8972a21f734fe035513b6fafe615",
    }
    env_map: Dict[str, str] = {
        "codellama-7b": "EVAL_OPEN_QUANT_CODELLAMA_7B",
        "codellama-34b": "EVAL_OPEN_QUANT_CODELLAMA_34B",
        "Wizardcoder33b": "EVAL_OPEN_QUANT_WIZARD_33B",
    }
    if eval_model not in env_map:
        return None
    env_key = env_map[eval_model]
    if eval_model in ("codellama-7b"):
        return os.environ.get(env_key) or None
    return os.environ.get(env_key) or defaults[eval_model]


def replicate_run_opensource_quantized(
    preprompt: str, prompt: str, eval_model: str
) -> str:
    """
    Generates text using the **quantized** variant (Replicate) for CodeLlama / WizardCoder.

    7B/13B: without ``EVAL_OPEN_QUANT_CODELLAMA_7B`` / ``_13B``, falls back to the same
    ``models.Codellama*`` used in ``eval.py`` (``meta/`` weights, **not** GGUF) to avoid
    breaking the eval pipeline.
    """
    model_id = get_quantized_replicate_model_id(eval_model)
    if not model_id:
        if eval_model == "codellama-7b":
            logger.warning(
                "EVAL_OPEN_QUANT_CODELLAMA_7B not set; falling back to models.Codellama7b (non-quantized on Replicate)."
            )
            return models.Codellama7b(preprompt, prompt)
        logger.error("No quantized Replicate ID mapped for: %s", eval_model)
        return ""

    return models._replicate_run_stream(
        model_id,
        _replicate_opensource_std_input(preprompt, prompt),
    )


class CustomFormatter(logging.Formatter):
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    blue = "\x1b[34:20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("devs-eval")  # get logger


# https://stackoverflow.com/questions/12507206/how-to-completely-traverse-a-complex-dictionary-of-unknown-depth
def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]


def rag_knowledge(Retriever, query):
    knowledge = ""
    questions = Retriever.generate_prompt_for_index(query)
    context = Retriever.query_documents(questions)
    for i, c in enumerate(context):
        if i >= 4:
            break
        knowledge += f"Context {i}: \n"
        knowledge += c
    return knowledge


# remove unwanted text for output
def remove_unwanted_characters(text):
    if text is None:
        return None

    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    text = ansi_escape.sub("", text)
    unwanted_pattern = re.compile(r"[^\x00-\x7F]+")  # Non-ASCII characters
    text = unwanted_pattern.sub("", text)

    return text


def delete_all_files_in_directory(folder):
    if not os.path.isdir(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))



def set_google_credentials():
    """Gemini — uses GOOGLE_API_KEY (Google AI Studio), not OpenAI or Replicate."""
    if "GOOGLE_API_KEY" not in os.environ:
        gemini_secret_access_key = getpass.getpass(
            "Google API Key (Gemini / Google AI): "
        )
        os.environ["GOOGLE_API_KEY"] = gemini_secret_access_key


def set_replicate_credentials():
    """Codellama / WizardCoder on Replicate — separate API key from OpenAI and Google."""
    if "REPLICATE_API_TOKEN" not in os.environ:
        replicate_secret_access_key = getpass.getpass(
            "Replicate API Key (codellama/wizardcoder): "
        )
        os.environ["REPLICATE_API_TOKEN"] = replicate_secret_access_key


def set_huggingface_credentials():
    # For gemini:
    if "HF_API_TOKEN" not in os.environ:
        hf_secret_access_key = getpass.getpass(
            "Enter Huggingface API Token (for use of various models): "
        )
        os.environ["HF_API_TOKEN"] = hf_secret_access_key


# find each subdirectory
def list_all_subdirectories_and_eval(
    data_dir, base_eval_dir, final_eval_dir, PROMPT_ENHANCEMENT_STRAT, Retriever
):
    data_root = Path(data_dir).resolve()
    for dirpath, _, filenames in os.walk(data_root):
        dirpath_p = Path(dirpath).resolve()
        try:
            rel = dirpath_p.relative_to(data_root)
        except ValueError:
            continue
        # Windows: never use data_dir + "/" — it does not match backslashes and breaks destination paths.
        subdir = "" if rel == Path(".") else str(rel)
        create_evaluation_directories(subdir, base_eval_dir=base_eval_dir, final_eval_dir=final_eval_dir)
        for file in filenames:
            if file.endswith(".csv"):
                file_path = os.path.abspath(os.path.join(dirpath, file))
                print(file_path)
                # Perform evaluation:
                for model in EVAL_MODELS:
                    # Note: Do not overwrite existing files, and do not evaluate if file exists already
                    eval_filepath, final_file_path, file_exists, NUM_EXISTING_SAMPLES = (
                        copy_csv_to_evaluation(
                            file_path,
                            subdir,
                            model,
                            PROMPT_ENHANCEMENT_STRAT,
                            base_eval_dir=base_eval_dir,
                            final_eval_dir=final_eval_dir,
                        )
                    )
                    if file_exists:
                        continue
                    read_models(
                        model,
                        PROMPT_ENHANCEMENT_STRAT,
                        NUM_EXISTING_SAMPLES,
                        eval_filepath,
                        final_file_path,
                        Retriever,
                    )


# function to create evaluation directories based on a given subdirectory
def create_evaluation_directories(subdir, base_eval_dir="evaluation/tmp", final_eval_dir="results"):
    for model in EVAL_MODELS:
        # construct the new directory path
        eval_dir_path = os.path.join(base_eval_dir, model, subdir)
        final_eval_dir_path = os.path.join(final_eval_dir, model, subdir)
        # create the directory if it does not exist
        os.makedirs(eval_dir_path, exist_ok=True)
        os.makedirs(final_eval_dir_path, exist_ok=True)


def make_column_names_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            dup + "." + str(i) if i != 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    # print(cols.tolist())
    # while True:
    #     x=1
    return df


def fix_duplicate_columns(dest_file_path):
    """
    Deduplicated csv is written to original file path
    """
    # FIX?: this is ignoring the header in the original file and manually extracting the first line???
    df = pd.read_csv(dest_file_path, header=None)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    if not df.columns.is_unique:  # First check if there are duplicate columns:
        df = make_column_names_unique(df)
        df.to_csv(dest_file_path, index=False, encoding="utf-8")
        logger.info(
            f"Evaluation file {dest_file_path} had duplicate columns, deduplicated them."
        )


def determine_eval_samples(dest_file_path):
    """
    Determine the number of samples currently present in a given evaluated dataset file.
    Also determines columns to remove (i.e., which are empty, because a previous evaluation run was not complete, which can only occur if copy_csv was successful but read_models was interrupted)
    Note: this is a variant of the same function used in llm-judge-eval.py
    """

    drop_cols = []

    df = pd.read_csv(dest_file_path, header=None)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    num_samples = 0
    for col in df.columns:
        if "LLM Correct?" in col:
            if not pd.isnull(
                df[col].iloc[0]
            ):  # this means that the column is not empty (i.e., prev evaluation passed through successfully)
                num_samples += 1
            else:
                cols_to_drop = list(_EVAL_SAMPLE_COL_BASES)
                drop_cols.extend(
                    [col_base + str(col.split("#")[1]) for col_base in cols_to_drop]
                )

    return num_samples, drop_cols


# function to copy CSV to the new evaluation directory and rename it
def copy_csv_to_evaluation(
    src_file_path,
    subdir,
    model,
    PROMPT_ENHANCEMENT_STRAT,
    base_eval_dir="evaluation/tmp",
    final_eval_dir="results",
):
    # FIX?: this should be passed in as Path? if we are using Path, we should use it outside instead of os.
    # dataset_file_name = os.path.basename(src_file_path)
    dataset_file_name = Path(
        src_file_path
    ).stem  # https://stackoverflow.com/questions/678236/how-do-i-get-the-filename-without-the-extension-from-a-path-in-python
    eval_filename_prefix = "evaluation-dataset-for-{}".format(dataset_file_name)
    # eval_filename = eval_file_name_prefix + "-" + dataset_file_name

    eval_filename = (
        "{}-{}.csv".format(eval_filename_prefix, PROMPT_ENHANCEMENT_STRAT)
        if PROMPT_ENHANCEMENT_STRAT != ""
        else "{}.csv".format(eval_filename_prefix)
    )

    dest_file_path = os.path.join(base_eval_dir, model, subdir, eval_filename)

    final_file_path = os.path.join(final_eval_dir, model, subdir, eval_filename)

    df = pd.read_csv(src_file_path, header=None)
    # set the third row as the header
    # FIX?: if this is the case for every file, why are they even saved?

    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    # reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)

    num_eval_samples = 0  # current eval samples in the evaluated dataset file.

    # Skip if file already exists and contains enough samples:
    if os.path.isfile(dest_file_path):
        fix_duplicate_columns(dest_file_path)
        num_eval_samples, drop_cols = determine_eval_samples(dest_file_path)
        if num_eval_samples >= NUM_SAMPLES_PER_TASK:
            logger.info(
                f"Skipping evaluation for {dest_file_path} as file already exists, and it has enough samples."
            )
            return dest_file_path, None, True, num_eval_samples

        logger.info(
            f"Evaluation file {dest_file_path} already exists, but has not enough samples (required: {NUM_SAMPLES_PER_TASK}, existing: {num_eval_samples}), will continue evaluation."
        )
        # Replace df with existing evaluated dataset file df:
        # FIX?: this is ignoring the header in the original file and manually extracting the first line??? why?????
        df = pd.read_csv(dest_file_path, header=None)
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        # reset the index of the DataFrame
        df.reset_index(drop=True, inplace=True)

        if len(drop_cols) > 0:
            df.drop(columns=drop_cols, inplace=True, errors="ignore")

    logger.info(f"Performing evaluation on {dest_file_path}.")
    # read the rest of the file starting from the fourth row without a header

    nrows = len(df)
    to_add = {}
    for i in range(num_eval_samples, NUM_SAMPLES_PER_TASK):
        for col_base in _EVAL_SAMPLE_COL_BASES:
            c = col_base + str(i)
            to_add[c] = [""] * nrows
    if to_add:
        df = pd.concat(
            [df, pd.DataFrame(to_add, index=df.index, dtype=object)], axis=1
        )

    df.to_csv(dest_file_path, index=False, encoding="utf-8")

    return dest_file_path, final_file_path, False, num_eval_samples


# gpt result
def read_models(
    model,
    PROMPT_ENHANCEMENT_STRAT,
    NUM_EXISTING_SAMPLES,
    eval_filepath,
    final_filepath,
    Retriever,
):
    # read the first four lines to determine the header

    with open("prompt_templates/system-prompt.txt", "r") as file2:
        preprompt = file2.read()

    # Read from evaluation dataset file:
    df = pd.read_csv(eval_filepath, header=0)
    for _c in df.columns:
        if "LLM " in str(_c) and " #" in str(_c):
            df[_c] = df[_c].astype(object)

    for index, row in df.iterrows():
        # iterate every row
        # find specific column
        model_evaluation(
            row,
            preprompt,
            df,
            index,
            model,
            PROMPT_ENHANCEMENT_STRAT,
            NUM_EXISTING_SAMPLES,
            Retriever,
        )
        df.to_csv(eval_filepath, index=False, encoding="utf-8")
        df.to_csv(final_filepath, index=False, encoding="utf-8")

    logger.info(f"Finished evaluation for {model}")




def semantic_judge_for_model(model: str, user_message: str) -> str:
    """
    Semantic judge: uses the same backend as the model under evaluation (or the closest available).
    Receives only the user message; the system prompt is SEMANTIC_JUDGE_SYSTEM.
    """
    preprompt = SEMANTIC_JUDGE_SYSTEM
    if model == "gpt-5.3-codex":
        return models.GPT4(preprompt, user_message, gpt_client)
    if model == "gpt-5.4":
        return models.GPT3_5(preprompt, user_message, gpt_client)
    if model == "gemini-2.5-flash":
        return models.gemini(preprompt, user_message, model)
    if model == "codellama-7b":
        return replicate_run_opensource_quantized(preprompt, user_message, model)
    if model == "codellama-34b":
        return replicate_run_opensource_quantized(preprompt, user_message, model)
    if model == "Wizardcoder33b":
        return replicate_run_opensource_quantized(preprompt, user_message, model)
    logger.error("Semantic judge not mapped for model: %s", model)
    return '{"valid": false, "reason": "Judge model not configured"}'


def prompt_enhancements(prompt, PROMPT_ENHANCEMENT_STRAT, Retriever):
    if PROMPT_ENHANCEMENT_STRAT == "RAG":
        knowledge = rag_knowledge(Retriever, prompt)
        prompt = prompt_templates.RAG_prompt(knowledge, prompt)
    elif PROMPT_ENHANCEMENT_STRAT == "COT":
        prompt = prompt_templates.CoT_prompt(prompt)
    elif PROMPT_ENHANCEMENT_STRAT == "FSP":
        prompt = prompt_templates.FSP_prompt(prompt)
    else:
        prompt = "Here is the actual prompt: " + prompt
    return prompt


def model_evaluation(
    row,
    preprompt,
    df,
    index,
    model,
    PROMPT_ENHANCEMENT_STRAT,
    NUM_EXISTING_SAMPLES,
    Retriever,
):
    """
    Note: Multi-turn implies 2 turns only
    """
    prompt = row["prompt"]

    # Skip empty rows
    if isinstance(row["prompt"], float):
        if math.isnan(row["prompt"]):
            return

    prompt = prompt_enhancements(prompt, PROMPT_ENHANCEMENT_STRAT, Retriever)

    policy_file = row["canonical_solution"]
    num_correct = 0
    logger.info(f"Begin testing model: {model}")
    for i in range(NUM_EXISTING_SAMPLES, NUM_SAMPLES_PER_TASK):
        multi_turn_count = 1
        while True:
            is_empty_code = False
            logger.info(f"Sample {i} for model {model}")
            logger.info(f"Preprompt: {preprompt}")
            logger.info(f"Prompt: {prompt}")
            df.at[index, "LLM User Prompt #" + str(i)] = prompt
            df.at[index, "LLM System Prompt #" + str(i)] = preprompt
            if model == "gpt-5.3-codex":
                text = models.GPT4(preprompt, prompt, gpt_client)
            elif model == "gpt-5.4":
                text = models.GPT3_5(preprompt, prompt, gpt_client)
            elif model == "gemini-2.5-flash":
                text = models.gemini(preprompt, prompt, model)
            elif model == "codellama-7b":
                text = replicate_run_opensource_quantized(preprompt, prompt, model)
            elif model == "codellama-34b":
                text = replicate_run_opensource_quantized(preprompt, prompt, model)
            elif model == "Wizardcoder33b":
                text = replicate_run_opensource_quantized(preprompt, prompt, model)
            else:
                logger.error(f"Unknown model: {model}")
                err_msg = f"[eval error: unknown model {model!r}]"
                df.at[index, "LLM Output #" + str(i)] = err_msg
                df.at[index, "LLM Plannable? #" + str(i)] = False
                df.at[index, "LLM Correct? #" + str(i)] = "Failure"
                df.at[index, "LLM Notes #" + str(i)] = err_msg
                break

            text = text if text is not None else ""
            if not str(text).strip():
                _pt, _ = empty_response_log_note(model)
                logger.error(
                    "Empty model response (not a ``` delimiter issue). %s",
                    _pt,
                )
            else:
                logger.info(f"Model raw output: {text}")

            answer, code = separate_answer_and_code(text, DELIMITERS)
            if code == "":
                if str(text).strip():
                    logger.error("Error: Answer contains no code, skipping eval_pipeline.")
                is_empty_code = True
            logger.info("Answer is: {}".format(answer))
            logger.info("Code is: {}".format(code))

            df.at[index, "LLM Output #" + str(i)] = text
            if is_empty_code:
                if not str(text).strip():
                    _, _empty_note = empty_response_log_note(model)
                else:
                    _empty_note = "No fenced DNL/SES or detectable DEVS line in model output"
                x = empty_code_error(_empty_note)
            else:
                x = eval_pipeline(
                    code,
                    policy_file,
                    row["prompt"],
                    semantic_judge_fn=lambda msg: semantic_judge_for_model(
                        model, msg
                    ),
                )
            df.at[index, "LLM Plannable? #" + str(i)] = x["devs_plan_success"]
            df.at[index, "LLM Correct? #" + str(i)] = x["opa_evaluation_result"]
            df.at[index, "LLM Notes #" + str(i)] = x["notes"]
            logging.info("Plan Result Summary:")
            for key, value in x.items():
                logging.info(f"{key}: {value}")

            if PROMPT_ENHANCEMENT_STRAT == "multi-turn":
                if multi_turn_count == 2:  # only do 2 turns
                    break
                multi_turn_count += 1
                if code == "":
                    continue
                preprompt = prompt_templates.multi_turn_system_prompt()
                if not x["devs_plan_success"]:
                    prompt = prompt_templates.multi_turn_plan_error_prompt(
                        row["prompt"], code, x["devs_plan_error"]
                    )
                elif x["devs_plan_success"] == "Failure":
                    prompt = prompt_templates.multi_turn_rego_error_prompt(
                        row["prompt"], code, policy_file, x["devs_plan_error"]
                    )
                continue
            else:
                break


def read_eval_models(ctx: click.Context, param, value) -> List[str]:
    """
    Normalizes the model list. Avoids the bug where default=list + type=str becomes
    a textual repr and split(',') produces names like \"['gpt-5.4'\".
    """
    if value is None:
        return list(EVAL_MODELS)
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return list(EVAL_MODELS)
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [p.strip() for p in s.split(",") if p.strip()]


def read_config_file(path: Path) -> Tuple[int, List[str]]:
    try:
        with path.open("r") as file:
            config = json.load(file)
            return config.get("samples", NUM_SAMPLES_PER_TASK), config.get(
                "models", EVAL_MODELS
            )
    except BaseException:
        print("Invalid config file.", file=sys.stderr)
        sys.exit(1)


def set_logger(log_file: Path):
    """
    Set global logger with log file
    """
    logger.setLevel(logging.DEBUG)
    # Create parent directories if they don't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Setup File handler: https://stackoverflow.com/a/24507130/13336187
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(CustomFormatter())
    file_handler.setLevel(logging.DEBUG)
    # Setup Stream Handler (i.e. console)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    # Log to both file and console:
    logger.addHandler(ch)
    logger.addHandler(file_handler)


def setup_gpt_client():
    global gpt_client
    global embeddings_model
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter OpenAI API key:")
        os.environ["OPENAI_API_KEY"] = api_key

    api_key = os.environ["OPENAI_API_KEY"]

    gpt_client= OpenAI(api_key=api_key)



@click.command()
@click.option(
    "--samples",
    "-s",
    type=int,
    help="Number of samples per task.",
    default=NUM_SAMPLES_PER_TASK,
)
@click.option(
    "--quick-test",
    "-q",
    "quick_test",
    is_flag=True,
    help="Perform quick evaluation on only 2 rows within the main dataset.",
    default=False,
)
@click.option(
    "--models",
    "-m",
    type=str,
    help=(
        "Comma-separated models (e.g.: gpt-5.4,gemini-2.5-flash). "
        f"Available: {' '.join(EVAL_MODELS)}"
    ),
    callback=read_eval_models,
    default=",".join(EVAL_MODELS),
)
@click.option(
    "--config",
    "--file",
    "-c",
    "-f",
    type=click.Path(path_type=Path, exists=True),
    help="Path to config file for command line options.",
)
@click.option(
    "--log-file",
    "-l",
    "log_file",
    type=click.Path(path_type=Path),
    help="Path to log file.",
    default=DEFAULT_LOG_FILE,
)
@click.option(
    "--enhance-strat",
    "-e",
    "enhance_strat",
    type=click.Choice(PROMPT_ENHANCEMENT_STRATS),
    help=f"Prompt enhancement strategy. Available strategies: {' '.join(PROMPT_ENHANCEMENT_STRATS)}",
    default="",
)
# @click.argument("enhance_strat", nargs=1, type=str, default="")
def main(
    samples: int, models: List[str], config: Path, log_file: Path, enhance_strat: str, quick_test: bool
):
    """
    Evaluate models.
    Available enhancement strategy: "RAG", "COT", "FSP", or "multi-turn".
    Config file takes precedence over command line options.
    """  # FIX

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass

    if config is not None:
        samples, models = read_config_file(config)

    # changing config variables basing on command line options
    global NUM_SAMPLES_PER_TASK
    NUM_SAMPLES_PER_TASK = samples
    global EVAL_MODELS
    EVAL_MODELS = models

    set_logger(log_file)

    print(samples)
    print(models)

    PROMPT_ENHANCEMENT_STRAT = (
        enhance_strat
        # FIX?: should this be changed to an option instead of argument
    )

    # API keys per provider (do not mix: OpenAI ≠ Google ≠ Replicate)
    if MODELS_USING_REPLICATE.intersection(models):
        set_replicate_credentials()

    if "gemini-2.5-flash" in models:
        set_google_credentials()

    if "gpt-5.4" in models or "gpt-5.3-codex" in models:
        setup_gpt_client()

    if "Magicoder_S_CL_7B" in models:
        setup_magicoder_params()

    # Setup retriever:
    if "RAG" in PROMPT_ENHANCEMENT_STRAT:
        Retriever = llama_index_retriever.Retriever(
            stored_index="../retriever/aws-index",
            path="../docs/r",
        )
    else:
        Retriever = None

    # Import dataset:
    if not quick_test:
        dataset.import_dataset()
    else:
        dataset.import_dataset(quick_test=True)

    # specify the directory you want to search
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    base_eval_dir = os.path.join(
        data_dir, "..", "evaluation/tmp"
    )  # should be the same as script_dir
    final_eval_dir = os.path.join(data_dir, "..", "evaluation/results")
    # Create evaluation directories for each data directory
    # and perform model evaluation:
    list_all_subdirectories_and_eval(
        data_dir, base_eval_dir, final_eval_dir, PROMPT_ENHANCEMENT_STRAT, Retriever
    )


if __name__ == "__main__":
    main()