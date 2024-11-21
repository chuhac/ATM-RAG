#!/usr/bin/env python3
import pathlib
from copy import deepcopy
from typing import List, Optional, Tuple, Type, TypeVar

from pydantic.dataclasses import dataclass

PROMPTS_ROOT = (pathlib.Path(__file__).parent / "prompts").resolve()



def get_prompt(prompt_name, data_row):
    prompt_template = get_prompt_template(prompt_name)
    result = prompt_template.format_map(data_row)
    
    return result 


def get_prompt_template(prompt_name):
    template_path = PROMPTS_ROOT / (prompt_name + '.prompt')
    with open(template_path) as f:
        prompt_template = f.read().rstrip("\n")
    
    return prompt_template 

