import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from sklearn.metrics import classification_report

class SavePeftModelCallback(TrainerCallback):

    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir,
                                         f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)
        return control


class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        lm_logits = None
        if hasattr(model, 'v2_regularization') and model.v2_regularization > 0:
            disable_adapter = hasattr(model, 'base_model')  # peft
            if disable_adapter:
                with model.disable_adapter():
                    outputs = model(**inputs, return_lm_only=True)
                    lm_logits = outputs.logits
            else:
                outputs = model(**inputs)
                lm_logits = outputs.logits
        output = model(lm_logits=lm_logits, **inputs)
        if isinstance(output, tuple):
            _, token_classification_output = output
            loss = token_classification_output.loss
        else:
            loss = output.loss

        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        model.eval()
        model.config.use_cache = True  # faster
        labels = inputs.pop('labels')
        with torch.no_grad():
            lm_logits = None
            if hasattr(model, 'v2_regularization') and model.v2_regularization > 0:  # regularization for nado
                disable_adapter = hasattr(model, 'base_model')  # peft
                if disable_adapter:
                    with model.disable_adapter():
                        outputs = model(**inputs, return_lm_only=True)
                        lm_logits = outputs.logits
                else:
                    outputs, _ = model(**inputs)
                    lm_logits = outputs.logits
            output = model(lm_logits=lm_logits, **inputs)
            if isinstance(output, tuple):
                _, token_classification_output = output
                logits = token_classification_output.logits
            else:
                logits = output.logits
            # get pool_method
            if hasattr(model, 'base_model'):  # peft
                is_peft = True
                pool_method = getattr(model.base_model, 'pool_method', False)
            else:
                is_peft = False
                pool_method = getattr(model, 'pool_method', False)

            if pool_method == 'inflection':
                if is_peft:
                    labels, pool_idx, _ = model.base_model._get_inflection_position(labels)
                else:
                    labels, pool_idx, _ = model._get_inflection_position(labels)
                assert len(logits.shape) == 3 and logits.shape[-1] == 2

            if len(logits.shape) == 2:
                # pooling is applied
                if model.config.is_encoder_decoder:
                    seqlen = (labels != -100).sum(dim=1)
                else:
                    seqlen = torch.ne(inputs['input_ids'], model.config.pad_token_id).sum(-1)
                labels = labels[range(labels.shape[0]), seqlen - 1]
                eval_loss = F.cross_entropy(logits, labels)
                # compute metrics later
                if hasattr(model, 'v2') and model.v2:
                    idx = inputs['input_ids'][range(labels.shape[0]), seqlen - 1]
                    preds = torch.sigmoid(logits[range(logits.shape[0]), idx]) > 0.5
                    preds = preds.long()
                else:
                    preds = logits.argmax(-1)
                return eval_loss, preds, labels

            if logits.shape[1] == labels.shape[1] - 1:  # happens for v2
                labels = labels[:, 1:]
            if logits.shape[2] == 1: # binary classification # v2
                eval_loss = token_classification_output.loss
                if pool_method == 'last':
                    if model.config.is_encoder_decoder:
                        seqlen = (labels != -100).sum(dim=1)
                    else:
                        seqlen = torch.ne(inputs['input_ids'], model.config.pad_token_id).sum(-1)
                    labels = labels[range(labels.shape[0]), seqlen - 1]
                    breakpoint()
                    return eval_loss, logits[:, -1], labels
            else:
                eval_loss = F.cross_entropy(logits.permute(0, 2, 1), labels)
        if prediction_loss_only:
            return eval_loss
        # loss, logit, label
        # NOTE: for efficiency, compute accuracy here!
        label_mask = labels != -100  # TODO -100 is pad idx by default;
        if logits.shape[2] == 1: # binary classification
            preds = torch.sigmoid(logits.squeeze(2)) > 0.5
        else:
            preds = logits.argmax(-1)
        # 1. accuracy
        # NOTE: no need to care about padding since preds cannot be -100
        correct = preds == labels
        true_acc = correct.sum(-1) / label_mask.sum(-1)  # [B,]
        # 2. prec, recall, f1
        tp, fp, fn = [], [], []
        for pred, label, mask in zip(preds, labels, label_mask):
            pred = pred[mask]
            label = label[mask]
            tp.append(((pred == 1) & (label == 1)).sum())
            fp.append(((pred == 1) & (label != 1)).sum())
            fn.append(((pred != 1) & (label == 1)).sum())
        tp = torch.stack(tp)
        fp = torch.stack(fp)
        fn = torch.stack(fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # nan handling before f1
        precision = torch.nan_to_num(precision, nan=0.0)
        recall = torch.nan_to_num(recall, nan=0.0)
        f1 = (2 * precision * recall) / (precision + recall)
        # handle nan for f1 once again
        f1 = torch.nan_to_num(f1, nan=0.0)

        metrics = torch.cat([met.unsqueeze(1) for met in (true_acc, precision, recall, f1)], dim=1)

        dummy_label = torch.full((labels.shape[0],), -100)
        return (eval_loss, metrics, dummy_label)


def compute_metrics(eval_preds):
    """Compute accuracy"""
    if not eval_preds.label_ids[0] == -100:
        # compute metrics here
        metrics = classification_report(eval_preds.label_ids, eval_preds.predictions, output_dict=True)
        return dict(accuracy=metrics['accuracy'],
                    precision=metrics['1']['precision'],
                    recall=metrics['1']['recall'],
                    f1=metrics['1']['f1-score'])
    metrics = eval_preds.predictions  # [N, 4]
    final_metrics = metrics.mean(0).tolist()

    return dict(accuracy=final_metrics[0],
                precision=final_metrics[1],
                recall=final_metrics[2],
                f1=final_metrics[3])
