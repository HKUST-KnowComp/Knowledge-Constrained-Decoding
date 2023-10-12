import os
import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from kcd.token_classifier.model import T5DoubleHeadModel

ENCODER_DECODER_ARCH_NAMES = ['t5', 't0', 'ul2', 'bart']


def load_transformer_LM_tokenizer(model_name_or_path,
                                  tokenizer_name_or_path=None,
                                  load_t5_doublehead=False,
                                  **kwargs):
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if load_t5_doublehead:
        model = T5DoubleHeadModel.from_pretrained(model_name_or_path, **kwargs)
    elif any(name in model_name_or_path.lower() for name in ENCODER_DECODER_ARCH_NAMES):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        # open-ended generation
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        # this is needed since we are using batched generation for causal LM
        tokenizer.padding_side = 'left'
    return model, tokenizer


def shift_right(input_ids, decoder_start_token_id):
    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    return shifted_input_ids


def get_logger(fname="logs/fever_test.log"):
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger(__name__)

    fileHandler = logging.FileHandler(fname)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger


def logsumexp(tensor, dim=-1, mask=None):
    if mask is None:
        return torch.logsumexp(tensor, dim=dim)

    assert mask.shape == tensor.shape, 'The factors tensor should have the same shape as the original'
    # a = torch.cat([torch.max(tensor, dim, keepdim=True) for _ in range(tensor.shape[dim])], dim)
    a = tensor.max(dim, keepdim=True)
    return a + torch.sum((tensor - a).exp() * mask, dim).log()


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def set_random_seeds(seed):
    """
    set the random seed of all related libraries
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
