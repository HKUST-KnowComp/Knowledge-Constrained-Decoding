import os
from dataclasses import dataclass, field

import torch
from bitsandbytes.optim import AdamW8bit
from dataloader import DataCollatorForSeq2SeqTokenClassification, load_data
from kcd.util import freeze_module
from model import T5EncoderForTokenClassification, T5ForTokenClassification, T5DoubleHeadModel
from kcd.attribute_classifier.attribute_classifier_model import AttributeClassifier
from torch.optim import AdamW
from trainer import MyTrainer, compute_metrics, SavePeftModelCallback
from transformers import (AutoTokenizer, T5ForConditionalGeneration,
                          DataCollatorForTokenClassification, DataCollatorForSeq2Seq,
                          HfArgumentParser, TrainingArguments,
                          get_linear_schedule_with_warmup, BitsAndBytesConfig)
from peft import (LoraConfig, PeftModel, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
import wandb

ENCODER_DECODER_ARCH_NAMES = ['t5', 't0', 'bart', 'blenderbot']


@dataclass
class ExperimentArgs(TrainingArguments):
    # training args
    evaluation_strategy: str = 'steps'
    dataloader_drop_last: bool = False
    report_to: str = 'wandb'
    save_strategy: str = 'steps'
    dataloader_num_workers: int = 0
    gradient_checkpointing: bool = False
    remove_unused_columns: bool = False  # keep to False
    ######################## Experiment Args ###########################
    # model args
    model_name: str = 'google/flan-t5-xxl'
    is_decoder: bool = True
    is_encoder_decoder: bool = False
    num_labels: int = 2
    pool_method: str = 'none'  # choices=['none', 'last', 'random', 'mean']
    classifier_dropout: float = None
    only_classifier: bool = False
    use_mlp_classifier: bool = False
    v2: bool = False
    nado_reg: float = 0.0
    # wandb logging
    wandb_project_name: str = "Knowledge Discriminator"
    wandb_run_name: str = 'KnowDisc'
    # peft
    use_lora: bool = field(
        default=False,
        metadata={"help": "use lora with huggingface peft. You must install loralib and peft."})
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    train_8bit: bool = False
    # checkpoint loading
    load_checkpoint: str = field(default=None)
    load_peft_checkpoint: str = field(default=None)
    load_classifier: str = field(default=None)
    # mode
    sft: bool = False
    test_only: bool = False
    # data
    dataset: str = 'fever'
    train_data_path: str = 'fever-train-kilt-processed.jsonl'
    validation_data_path: str = 'fever-dev-kilt-processed.jsonl'
    use_kilt_format: bool = True
    test_data_path: str = None
    sequence_label: bool = False

    use_pdb_debug: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.is_encoder_decoder:
            print("is_encoder_decoder is deprecated. Use is_decoder instead.")
            self.is_decoder = True
            self.is_encoder_decoder = False


def main():
    parser = HfArgumentParser([ExperimentArgs])
    args = parser.parse_args_into_dataclasses()[0]
    args.run_name = args.wandb_run_name
    os.environ['WANDB_PROJECT'] = args.wandb_project_name

    if args.local_rank in (-1, 0) and not args.use_pdb_debug:  # main process
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, anonymous='allow')

    # advanced training configs
    if args.deepspeed and args.train_8bit:
        raise ValueError("--train_8bit is not compatible with deepspeed.")
    if args.train_8bit:
        args.ddp_find_unused_parameters = False  # integral for train_8bit
    device_map = 'auto'

    load_kwargs = {
        'device_map': device_map,
        'num_labels': args.num_labels,
        'pool_method': args.pool_method,
        'classifier_dropout': args.classifier_dropout,
        'load_in_8bit': args.train_8bit,
        'torch_dtype': torch.float16 if args.train_8bit else torch.bfloat16,
        'v2': args.v2,
        'v2_regularization': args.nado_reg,
    }
    if args.use_mlp_classifier:
        load_kwargs['use_mlp_classifier'] = True
        load_kwargs.pop('load_in_8bit')
        load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=args.train_8bit,
                                                                llm_int8_skip_modules=['lm_head', 'mlp_layer1', 'mlp_layer2'])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.truncation_side = 'left'
    if any(name in args.model_name.lower() for name in ENCODER_DECODER_ARCH_NAMES):
        encoder_decoder = True
        modules_to_save = ['classifier']
        if args.sft:
            model_class = T5ForConditionalGeneration
            modules_to_save = ['lm_head']
            if args.is_encoder_decoder:
                target_modules = '.*(q|v|o)$'
            elif args.is_decoder:
                target_modules = '.*decoder.*(q|v|o)$'
            load_kwargs.pop('num_labels')
            load_kwargs.pop('pool_method')
            load_kwargs.pop('classifier_dropout')
            load_kwargs.pop('v2')
            load_kwargs.pop('v2_regularization')
        elif args.is_encoder_decoder:
            model_class = T5ForTokenClassification
            target_modules = '.*(q|v|o)$'
        elif args.is_decoder:
            print("for is_decoder, I will train T5 encoder decoder model with a"
                  " frozen encoder.")
            model_class = T5ForTokenClassification
            target_modules = '.*decoder.*(q|v|o)$'
        else:
            model_class = T5EncoderForTokenClassification
            target_modules = ['q', 'v', 'o']
    else:
        encoder_decoder = False
        load_kwargs.pop('load_in_8bit')
        load_kwargs.pop('classifier_dropout')
        load_kwargs.pop('v2')
        load_kwargs.pop('v2_regularization')
        args.train_8bit = False
        print("GPT2 uses conv layers, which are not supported by 8bit training.")
        model_class = AttributeClassifier
        modules_to_save = ['score']
        target_modules = ['c_proj', 'c_attn']

    if args.nado_reg > 0:  # need lm logit for regularization
        model_class = T5DoubleHeadModel
    model = model_class.from_pretrained(args.model_name, **load_kwargs)
    if not encoder_decoder:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if args.train_8bit and not args.test_only:
        model = prepare_model_for_int8_training(model)

    if args.only_classifier:
        print("only training last classifier layer; freezing all other layers")
        args.use_lora = False
        model.train()
        freeze_module(model)
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable_names = []
        num_total_params = 0
        for key, param in model.named_parameters():
            if param.requires_grad:
                trainable_names.append(key)
            num_total_params += 1
        print(f"trainable parameters: {trainable_names}, {len(trainable_names) / num_total_params * 100}%")


        if args.load_classifier:
            model.classifier.load_state_dict(torch.load(args.load_classifier), strict=True)

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if encoder_decoder else TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        if args.load_peft_checkpoint:
            model = PeftModel.from_pretrained(model, args.load_peft_checkpoint)
        else:
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if args.gradient_checkpointing and not args.train_8bit:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint), strict=True)

    datasets = load_data(args, tokenizer, model.config.is_encoder_decoder,
                         decoder_start_token_id=model.config.decoder_start_token_id)

    if args.sft:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    elif args.is_encoder_decoder or args.is_decoder:
        collator = DataCollatorForSeq2SeqTokenClassification(tokenizer)
    else:
        collator = DataCollatorForTokenClassification(tokenizer)

    if args.train_8bit:
        optimizer = AdamW8bit(model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * (
        len(datasets['train']) // args.train_batch_size)
    lr_decay = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=args.warmup_steps,
                                               num_training_steps=num_training_steps)
    trainer = MyTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_decay),
        callbacks=[SavePeftModelCallback] if args.use_lora else None,
    )

    if args.use_pdb_debug:
        def get_pred_label(split, index):
            inputs = {k: torch.LongTensor([v]).to('cuda') for k, v in datasets[split][index].items()}
            _, out = model(**inputs)
            print(inputs['labels'])
            print(out.logits.argmax(-1))
            return inputs, out
        breakpoint()

    if not args.test_only:
        trainer.train()
        # model.save_pretrained(f"{args.output_dir}")
    trainer.evaluate(datasets['validation'], metric_key_prefix="eval")
    if datasets['test'] is not None:
        trainer.evaluate(datasets['test'], metric_key_prefix="test")
    else:
        print("No test set is provided. Skip evaluation.")


if __name__ == '__main__':
    main()
