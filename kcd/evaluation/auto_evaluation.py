from pprint import pprint

from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore


def evaluate_per_sent(preds: list[str], targets: list[list[str]], metric: str, bleu_weights=None):
    all_scores = [evaluate(p, t, metrics=(metric,), bleu_weights=bleu_weights)[metric].item()
                  for p, t in zip(preds, targets)]
    return all_scores


def evaluate(preds, target, metrics=('bleu', 'rougeL'), bleu_weights=None):
    metric_fn = {}
    if 'bleu' in metrics:
        if bleu_weights is not None:
            bleu = BLEUScore(weights=bleu_weights)
        else:
            bleu = BLEUScore()
        metric_fn['bleu'] = bleu
    if 'rougeL' in metrics:
        rouge = ROUGEScore(rouge_keys="rougeL")
        metric_fn['rougeL'] = rouge

    scores = {}
    for metric, func in metric_fn.items():
        scores[metric] = func(preds, target)

    return scores


if __name__ == '__main__':
    preds = ["hello there", "general kenobi"]
    target = [["hello there", "hi there"], ["master kenobi", "general canopi"]]
    scores = evaluate(preds, target)
    pprint(scores)
