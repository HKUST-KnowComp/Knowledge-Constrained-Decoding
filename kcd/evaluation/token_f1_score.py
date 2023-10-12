"""
Taken from parlai.

Sehyun Choi, 2023
"""
import re
from collections import Counter
from typing import List

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


class TokenF1Score:
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute(guess: str, answers: List[str], expose_p_and_r: bool = False):
        g_tokens = normalize_answer(guess).split()
        scores = [
            TokenF1Score.prec_recall_f1_score(g_tokens,
                                              normalize_answer(a).split()) for a in answers
        ]
        max_p, max_r, max_f1 = 0, 0, 0
        for p, r, f1 in scores:
            max_p, max_r, max_f1 = max(max_p, p), max(max_r, r), max(f1, max_f1)
        if expose_p_and_r:
            return max_p, max_r, max_f1
        return max_f1

    @staticmethod
    def batch_compute(guesses: List[str], answers: List[List[str]], expose_p_and_r: bool = False):
        assert len(guesses) == len(answers)
        return [
            TokenF1Score.compute(guess, answer, expose_p_and_r=expose_p_and_r)
            for guess, answer in zip(guesses, answers)
        ]
