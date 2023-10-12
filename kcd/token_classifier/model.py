import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, LogSigmoid
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (T5EncoderModel, T5ForConditionalGeneration, T5Model,
                                                T5PreTrainedModel, T5Stack)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp_layer1 = nn.Linear(input_dim, hidden_dim)
        self.mlp_layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.mlp_layer1(x)))
        x = self.mlp_layer2(x)
        return x

    def _init_weights(self, factor):
        self.mlp_layer1.weight.data.normal_(mean=0.0, std=factor * 1.0)
        self.mlp_layer1.bias.data.zero_()
        self.mlp_layer2.weight.data.normal_(mean=0.0, std=factor * 1.0)
        self.mlp_layer2.bias.data.zero_()


class EncoderDecoderTokenClassificationMixin:

    def _get_inflection_position(self, labels, hidden_states=None):
        # TODO: v2 compatible
        eos_idx = (labels != -100).sum(dim=1) - 1
        inflect_idx = (labels == 1).sum(dim=1) - 1 # the last position of 1
        inflect_idx = torch.where(inflect_idx < 0, 0, inflect_idx)
        # if self.v2:
        #     inflect_idx -= 1
        #     eos_idx -= 1

        pool_idx = []
        for inf_i, eos_i in zip(inflect_idx, eos_idx):
            if inf_i == eos_i or inf_i == 0:  # this means no inflection point
                pos = torch.randint(0, eos_i, (1,))
            else:
                if torch.randn(1) > 0.5:
                    pos = inf_i.unsqueeze(-1)
                else:
                    pos = inf_i.unsqueeze(-1) + 1
            pos = pos.to(eos_i.device)
            pool_idx.append(torch.cat([pos, eos_i.unsqueeze(-1)]))
        pool_idx = torch.stack(pool_idx, dim=0)  # [B, 2]

        pooled_output = None
        if hidden_states is not None:  # [B, T, E]
            pooled_output = []
            for i in range(pool_idx.shape[0]):
                pooled = hidden_states[i, pool_idx[i]]  # [2, E]
                pooled_output.append(pooled)
            pooled_output = torch.stack(pooled_output, dim=0)  # [B, 2, E]

        # if self.v2:
        #     labels = torch.gather(labels, 1, pool_idx + 1)  # [B, 2]
        labels = torch.gather(labels, 1, pool_idx)  # [B, 2]

        return labels, pool_idx, pooled_output

    def _init_model(self,
                    config: T5Config,
                    num_labels=2,
                    classifier_dropout=None,
                    pool_method='none',
                    v2=False,
                    constraint_factor=1.0,
                    v2_regularization=0.0,
                    use_mlp_classifier=False):
        """v2: version 2 of token classifier: v2 is much more efficient."""
        self.pool_method = pool_method
        self.num_labels = num_labels
        self.v2 = v2
        self.constraint_factor = constraint_factor
        self.v2_regularization = v2_regularization
        self.return_lm_only = None

        classifier_dropout = (classifier_dropout
                              if classifier_dropout is not None else config.dropout_rate)
        self.dropout = nn.Dropout(classifier_dropout)

        if v2:
            self.num_labels = 1
            num_labels = config.vocab_size

        if use_mlp_classifier:
            self.classifier = MLP(config.d_model, config.d_model * 2, num_labels)
        else:
            self.classifier = nn.Linear(config.d_model, num_labels)

    def _init_weights(self, module):
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, (T5ForTokenClassification, T5DoubleHeadModel)) and hasattr(
                module, "classifier"):
            if isinstance(module.classifier, MLP):
                module.classifier._init_weights(factor)
            else:
                module.classifier.weight.data.normal_(mean=0.0, std=factor * 1.0)
                if hasattr(module.classifier, "bias") and module.classifier.bias is not None:
                    module.classifier.bias.data.zero_()

    def forward(self, return_dict=True, labels=None, output_hidden_states=True,
                return_lm_only=None, lm_logits=None, **kwargs):
        del return_dict, output_hidden_states
        if self.v2:
            decoder_input_ids = kwargs.get('decoder_input_ids', None)
            assert decoder_input_ids is not None
            decoder_attention_mask = kwargs.get('decoder_attention_mask', None)
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids != self.config.pad_token_id
                decoder_attention_mask[:, 0] = 1  # decoder_start_token_id
        outputs = super().forward(return_dict=True, output_hidden_states=True, **kwargs)
        if self.return_lm_only or return_lm_only:
            return outputs

        last_hidden_state = outputs.decoder_hidden_states[-1]  # decoder hidden state
        dropped = self.dropout(last_hidden_state)
        if not self.training:
            # TODO: bnb might have a bug; during eval mode, the input gets casted to
            # fp32, while the weight is fp16.
            if isinstance(self.classifier, MLP):
                dtype = self.classifier.mlp_layer1.weight.data.dtype
            elif hasattr(self.classifier, 'weight'):
                dtype = self.classifier.weight.data.dtype
            else:
                # dtype = torch.float16
                dtype = next(self.classifier.parameters()).data.dtype
            dropped = dropped.to(dtype)  # [B, T, H]

        if self.pool_method == 'none':
            pooled_output = dropped
        elif self.pool_method == 'last':
            if labels is not None:
                seqlen = (labels != -100).sum(dim=1)
                pool_idx = seqlen - 1
                if self.v2:
                    pool_idx -= 1
                pooled_output = dropped[range(dropped.shape[0]), pool_idx]
                labels = labels[range(labels.shape[0]), seqlen - 1]
            else:
                pool_idx = - 1
                if self.v2 and self.training:
                    pool_idx -= 1
                pooled_output = dropped[:, pool_idx, :]

        elif self.pool_method == 'inflection':
            # pool 2 positions: 0.5 * before | 0.5 * after, eos
            assert labels is not None
            labels, pool_idx, pooled_output = self._get_inflection_position(labels, hidden_states=dropped)

        elif self.pool_method == 'random':
            if labels is not None:
                seqlen = (labels != -100).sum(dim=1)
                if self.v2:
                    pool_idx = torch.cat([torch.randint(1, s, (1,)) - 1 for s in seqlen])
                else:
                    pool_idx = [torch.randint(0, s, (1,)) for s in seqlen]
                pooled_output = dropped[range(dropped.shape[0]), pool_idx]
                labels = labels[range(labels.shape[0]), seqlen - 1]
            else:
                if self.v2:
                    pool_idx = torch.randint(1, dropped.shape[1], (1,)) - 1
                else:
                    pool_idx = torch.randint(0, dropped.shape[1], (1,))
                pooled_output = dropped[:, pool_idx]
        else:
            raise ValueError(f'pool_method {self.pool_method} not supported.')

        class_logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.v2:
                if self.pool_method in ('none', 'inflection'):
                    if self.pool_method == 'none':
                        length = decoder_input_ids.shape[1]
                        index_iterator = range(length - 1)
                    else:
                        index_iterator = pool_idx.T  # [B, 2] -> [2, B]
                    all_pred_logits = []
                    for i in index_iterator:
                        pred_logits = torch.gather(class_logits[:, i, :], 1, decoder_input_ids[:, i + 1].unsqueeze(1))
                        loss_fct = BCEWithLogitsLoss(weight=decoder_attention_mask[:, i + 1].unsqueeze(1))
                        curr_loss = loss_fct(pred_logits, labels[:, i].unsqueeze(1).float())
                        if loss is not None:
                            loss += curr_loss
                        else:
                            loss = curr_loss
                        all_pred_logits.append(pred_logits)
                    pred_logits = torch.cat(all_pred_logits, dim=1)  # [B, T]
                    # regularization for nado
                    if self.v2_regularization > 0.0:
                        assert lm_logits is not None
                        final_logits = (torch.log_softmax(lm_logits, dim=-1) +
                                        nn.LogSigmoid()(class_logits) * self.constraint_factor)
                        if self.pool_method == 'none':
                            length = decoder_input_ids.shape[1]
                            index_iterator = range(1, length)
                        else:
                            index_iterator = pool_idx.T  # [B, 2] -> [2, B]
                        loss2 = 0
                        for i in index_iterator:
                            _pred_logits = torch.gather(class_logits[:, i - 1, :], 1, decoder_input_ids[:, i].unsqueeze(1))
                            pred_probs = torch.sigmoid(_pred_logits)
                            sum_log = torch.logsumexp(final_logits[:, i, :], dim=-1).unsqueeze(1)
                            sum_probs = torch.exp(sum_log)
                            sum_logits = torch.log(sum_probs / (1 - sum_probs))

                            loss_fct = BCEWithLogitsLoss(weight=decoder_attention_mask[:, i].unsqueeze(1))
                            loss2 += self.v2_regularization * (loss_fct(sum_logits, pred_probs) - loss_fct(_pred_logits, pred_probs))

                        loss += loss2
                else:
                    bs = class_logits.shape[0]
                    pred_logits = torch.gather(class_logits, 1, decoder_input_ids[range(bs), pool_idx + 1].unsqueeze(1))
                    loss_fct = BCEWithLogitsLoss(weight=decoder_attention_mask[range(bs), pool_idx + 1].unsqueeze(1))
                    loss = loss_fct(pred_logits, labels.unsqueeze(1).float())
                # during training, we return the pred_logits to eval with classification metrics
                class_logits = pred_logits.unsqueeze(-1)  # [B, T, 1]
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(class_logits.view(-1, self.num_labels), labels.view(-1))

        token_class_outs = TokenClassifierOutput(
            loss=loss,
            logits=class_logits,
        )

        return (outputs, token_class_outs)


class T5ForTokenClassification(EncoderDecoderTokenClassificationMixin, T5Model):
    """This is based on T5Model to save computing lm_head during training."""

    def __init__(self,
                 config: T5Config,
                 num_labels=2,
                 classifier_dropout=None,
                 pool_method='none',
                 v2=False,
                 v2_regularization=0.0,
                 use_mlp_classifier=False):
        super().__init__(config)
        self._init_model(config,
                         num_labels=num_labels,
                         classifier_dropout=classifier_dropout,
                         pool_method=pool_method,
                         v2=v2,
                         v2_regularization=v2_regularization,
                         use_mlp_classifier=use_mlp_classifier)
        self.post_init()


class T5DoubleHeadModel(EncoderDecoderTokenClassificationMixin, T5ForConditionalGeneration):
    """This is based on T5Model to save computing lm_head during training."""

    def __init__(self,
                 config: T5Config,
                 num_labels=2,
                 classifier_dropout=None,
                 pool_method='none',
                 v2=False,
                 v2_regularization=0.0,
                 use_mlp_classifier=False):
        super().__init__(config)
        self._init_model(config,
                         num_labels=num_labels,
                         classifier_dropout=classifier_dropout,
                         pool_method=pool_method,
                         v2=v2,
                         v2_regularization=v2_regularization,
                         use_mlp_classifier=use_mlp_classifier)

        self.post_init()


class T5DecoderForTokenClassification(T5PreTrainedModel):

    def __init__(self, config, num_labels=2, classifier_dropout=None, pool_method='none'):
        config.is_decoder = True
        config.is_encoder_decoder = False
        config.num_layers = config.num_decoder_layers
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = T5Stack(config, embed_tokens=self.shared)

        self.pool_method = pool_method
        self.num_labels = num_labels

        classifier_dropout = (classifier_dropout
                              if classifier_dropout is not None else config.dropout_rate)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.decoder.set_input_embeddings(new_embeddings)

    def forward(self, return_dict=True, labels=None, **kwargs):
        del return_dict
        outputs = self.decoder.forward(return_dict=True, **kwargs)

        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.dropout(last_hidden_state)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class T5EncoderForTokenClassification(T5EncoderModel):

    def __init__(self, config: T5Config, num_labels=2, classifier_dropout=None, pool_method='none'):
        super().__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.pool_method = pool_method

        classifier_dropout = (classifier_dropout
                              if classifier_dropout is not None else config.dropout_rate)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, return_dict=True, labels=None, **kwargs):
        del return_dict
        outputs = super().forward(return_dict=True, **kwargs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.dropout(last_hidden_state)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
