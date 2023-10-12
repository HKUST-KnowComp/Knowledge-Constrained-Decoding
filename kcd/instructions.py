BASE_INSTRUCTION_TEMPLATE = "{}\n\n{}"
ALPACA_INSTRUCTION_TEMPLATE = ("Below is an instruction that describes a task, "
                               "paired with an input that provides further context. "
                               "Write a response that appropriately completes the request.\n\n"
                               "### Instruction:\n{}\n\n"
                               "### Input:\n{}\n\n"
                               "### Response:")

OPENAI_INSTRUCTION = ("Use the following knowledge, but not directly copy, "
                      "to generate a concise response: \"{}\"")

TASK_INSTRUCTIONS = {
    'wow': {
        'instruction': "Given the dialog history and a relevant knowledge above,"
                       " generate a knowledgeable, useful, and helpful answer.",
        'input': "History:\n{}\n\nKnowledge:\n{}",
        'param': ['question', 'knowledge']
    },
    'fever': {
        'instruction': "Generate a claim that is entirely supported by the evidences above.",
        'input': "Evidences:\n{}",
        'param': ['knowledge']
    },
    'dstc11_task5': {
        'instruction': "Given the dialog history and a relevant knowledge above,"
                       " generate a knowledgeable, useful, and helpful answer.",
        'input': "History:\n{}\n\nKnowledge:\n{}",
        'param': ['question', 'knowledge']
    },
    'summarization': {
        'instruction': "Given the article above, generate a faithful summary.",
        'input': "### Document:\n{}",
        'param': ['knowledge']
    }
}

CLAIM_CLASSFICATION_INSTRUCTION = ("Given some evidences, determine whether the claim "
                                   "is supported by the evidences or not.\n\n"
                                   "### Claim:\n{}\n\n"
                                   "### Evidences:\n{}\n\n"
                                   "### Choices:\n- {}\n- {}")

WOW_CLASSFICATION_INSTRUCTION = ("Given the dialog history and a relevant knowledge, "
                                 "determine whether the response is supported by "
                                 "the knowledge or not."
                                 "### Knowledge:\n{}\n\n"
                                 "### History:\n{}\n\n"
                                 "### Response:\n{}\n\n"
                                 "### Choices:\n- Yes\n- No")

def get_instruction(model, task, **kwargs):
    if model == 'openai':
        if 'knowledge' not in kwargs:
            raise ValueError('Missing parameter: knowledge')
        return OPENAI_INSTRUCTION.format(kwargs.get('knowledge'))

    if task in ('cnn_dailymail', 'xsum'):
        task = 'summarization'
    components = TASK_INSTRUCTIONS[task]
    instruction_params = []
    for param in components['param']:
        if param not in kwargs:
            raise ValueError(f'Missing parameter: {param}')
        instruction_params.append(kwargs.get(param))

    inst_template = get_model_base_instruction(model,
                                               instruction=components['instruction'],
                                               input_text=components['input'])
    return inst_template.format(*instruction_params)


def get_model_base_instruction(model, instruction=None, input_text=None):
    if model == 'openai':
        return OPENAI_INSTRUCTION
    assert instruction is not None and input_text is not None
    if model == 'alpaca':
        return ALPACA_INSTRUCTION_TEMPLATE.format(instruction, input_text)
    return BASE_INSTRUCTION_TEMPLATE.format(input_text, instruction)
