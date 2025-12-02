ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Generate a docstring for this {} code

### Input:
{}

### Response:
{}"""


GRPO_PROMPT = """You are an expert software engineer who writes clear, comprehensive docstrings for code functions.

Task: Generate ONLY a docstring for the following {language} function.
The docstring should be a single paragraph that describes the function, its parameters, and its return value enclosed in triple quotes.

Code:
{code}

Docstring:"""
