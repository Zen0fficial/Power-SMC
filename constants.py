"""Shared prompt templates and constants for the power-SMC experiments."""

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np

# Generic math prompting templates.
PROMPT = "Can you solve the following math problem? "
BASE = " Put your final answer within \\boxed{{}}."
COT = " Please reason step by step, and put your final answer within \\boxed{{}}."
COT_ALT = " Please explain your reasoning with a detailed, step-by-step solution, and present your final answer within \\boxed{{}}."

# GPQA-specific instruction template.
GPQA_QUERY_TEMPLATE = "Answer the following multiple choice question. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). \n\n{Question} \n\nThink step by step before answering."
