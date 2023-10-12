import argparse
import logging
import os
import sys
import time
from typing import Dict

from datasets import load_dataset
import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup

# TODO: ugly fix of importing from parent folder's util.py :(
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from kcd.util import freeze_module, set_random_seeds
from kcd.attribute_classifier.attribute_classifier_model import AttributeClassifier
from kcd.attribute_classifier.attribute_dataloader import get_attribute_dataloader
from kcd.attribute_classifier.evaluate_classifier import evaluate


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="gpt2-large")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--max_length",
                        type=int,
                        default=50,
                        help="maximum length of complete sentence.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataname", type=str, default="sst2")
    parser.add_argument("--pool_method",
                        type=str,
                        default='last',
                        choices=['last', 'max', 'mean'],
                        help="choice of pooling of LM's hidden states over the time dimension.")

    parser.add_argument('--n_epochs', type=int, default=10, help='number of times to train')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=5e-5, help='the learning rate')
    parser.add_argument('--linear_scheduler', action='store_true')
    parser.add_argument('--decay_steps',
                        type=int,
                        default=0,
                        help='decay lr after x epochs. 0 means to use ReduceLrOnPlateau')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='how much to decay lr')
    # extra
    parser.add_argument('--seed', type=int, default=2022, help="the random seed")
    parser.add_argument('--verbose', action='store_true', help='whether to print results a lot')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='saved_models/',
                        help='where to save the model')
    args = parser.parse_args()

    return args


def train(model, dataloaders: Dict[str, torch.utils.data.DataLoader], args, device='cpu'):
    """ Trains a given model and dataset.

    obtained and adapted from:
    https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/master/src/train.py
    """

    # optimizers
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    momentum=0.9)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'{args.optimizer} not setup.')

    # lr schedulers
    if args.linear_scheduler:
        total_steps = len(dataloaders['train']) * args.n_epochs
        lr_decay = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=0,
                                                   num_training_steps=total_steps)
    elif args.decay_steps > 0:
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    else:
        lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=args.lr_decay_rate,
                                                              mode='max',
                                                              patience=30,
                                                              cooldown=20,
                                                              min_lr=1e-6,
                                                              verbose=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_valid_loss = np.inf
    best_valid_acc = 0

    since = time.time()
    for epoch in range(args.n_epochs):

        model.train()
        sample_count = 0
        running_loss = 0
        running_acc = 0
        epoch_valid_loss = None

        if args.verbose:
            logging.info(f'Epoch {epoch + 1}/{args.n_epochs}:\n')
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        for i, inputs in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = inputs.pop('labels')

            optimizer.zero_grad()
            yhat = model(**inputs, pool_method=args.pool_method).logits

            loss = loss_fn(yhat, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            sample_count += labels.size(0)
            running_loss += loss.item() * labels.size(0)  # smaller batches count less
            running_acc += (yhat.argmax(-1) == labels).sum().item()  # num corrects
            if isinstance(lr_decay, torch.optim.lr_scheduler.LambdaLR):
                lr_decay.step()

        epoch_train_loss = running_loss / sample_count
        epoch_train_acc = running_acc / sample_count

        # reduce lr
        if args.decay_steps > 0:
            lr_decay.step()
        elif not args.linear_scheduler:
            # reduce on plateau, evaluate to keep track of acc in each process
            epoch_valid_loss, epoch_valid_acc = evaluate(model,
                                                         dataloaders['valid'],
                                                         loss_fn,
                                                         args,
                                                         device=device)
            lr_decay.step(epoch_valid_acc)

        if args.verbose:  # only validate using process 0
            if epoch_valid_loss is None:  # check if process 0 already validated
                epoch_valid_loss, epoch_valid_acc = evaluate(model,
                                                             dataloaders['valid'],
                                                             loss_fn,
                                                             args,
                                                             device=device)

            logging.info(f'[Train] loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f} |'
                         f' [Valid] loss: {epoch_valid_loss:.4f} - acc: {epoch_valid_acc:.4f}')

            # save model and early stopping
            if epoch_valid_acc >= best_valid_acc:
                best_epoch = epoch + 1
                best_valid_acc = epoch_valid_acc
                best_valid_loss = epoch_valid_loss
                # saving using process (rank) 0 only as all processes are in sync
                save_name = os.path.join(args.checkpoint_dir, 'best.pth')
                torch.save(model.score.state_dict(), save_name)
            epoch_valid_loss = None  # reset loss

    if args.verbose:
        time_elapsed = time.time() - since
        logging.info(f'Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        saved_name = os.path.join(args.checkpoint_dir, 'best.pth')
        model.score.load_state_dict(torch.load(saved_name))  # load best model

        test_loss, test_acc = evaluate(model, dataloaders['test'], loss_fn, args, device=device)

        logging.info(f'Best [Valid] | epoch: {best_epoch} - loss: {best_valid_loss:.4f} '
                     f'- acc: {best_valid_acc:.4f}')
        logging.info(f'[Test] loss {test_loss:.4f} - acc: {test_acc:.4f}')


def main():
    args = options()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed != -1:
        set_random_seeds(args.seed)

    # output dir and logging setup
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(args.checkpoint_dir, 'train_log.log'), mode='a'),
        logging.StreamHandler(),
    ],
                        format='%(asctime)s:%(msecs)d|%(name)s|%(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.info('Start Training!')

    # Load pretrained model
    model = AttributeClassifier.from_pretrained(args.pretrained_model,
                                                output_hidden_states=True,
                                                resid_pdrop=0,
                                                embd_pdrop=0,
                                                attn_pdrop=0,
                                                summary_first_dropout=0,
                                                num_labels=args.num_labels)
    model.to(device)
    # Freeze GPT-2 weights; only train the classifer on top
    freeze_module(model.transformer)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    data = load_dataset(args.dataname)
    train_loader = get_attribute_dataloader(data,
                                            tokenizer,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            split='train',
                                            num_workers=0)
    valid_loader = get_attribute_dataloader(data,
                                            tokenizer,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            split='validation',
                                            num_workers=0)
    test_loader = get_attribute_dataloader(data,
                                           tokenizer,
                                           max_length=args.max_length,
                                           batch_size=args.batch_size,
                                           split='test',
                                           num_workers=0)

    dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    train(model, dataloaders, args, device=device)


if __name__ == '__main__':
    main()
