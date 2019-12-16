import torch
from torch import nn
import numpy as np
import argparse
import transformers
import random
import logging
import os
import models

from dataset import DatasetManager

ALL_MODELS = transformers.BertConfig.pretrained_config_archive_map.keys()

# Define paramters taken from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune BERT on WikiQA')
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The default batch size, influences GPU memory utilisation')
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use apex with fp16 weights and data')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Whether to use apex with fp16 weights and data')

    parser.add_argument('--train_file', type=str, required=True,
                        help='Cache directory for models')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Cache directory for models')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Cache directory for models')

    parser.add_argument('--epochs', default=20, type=int, 
                        help='Number of training epochs')
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Be quiet!')
    parser.add_argument('--seed', type=int, default=999,
                        help='Seed to reproduce training')
    parser.add_argument('--cache_dir', type=str, default='./_cache',
                        help='Cache directory for models')
    
    args = parser.parse_args()

    # Extract parameters
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    fp16 = args.fp16
    fp16_opt_level = args.fp16_opt_level
    epochs = args.epochs
    quiet = args.quiet
    seed = args.seed
    cache_dir = args.cache_dir
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    adam_epsilon = args.adam_epsilon
    train_file = args.train_file
    valid_file = args.valid_file
    test_file = args.test_file
    do_lower_case = args.do_lower_case
    model_name_or_path = args.model_name_or_path
    max_grad_norm = args.max_grad_norm

    # cache directory setup
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    # CUDA Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, 16-bits training: %s", device, fp16)

    # Seed setting for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ####### CREATE MODEL
    config = transformers.BertConfig.from_pretrained(model_name_or_path,
                                                     cache_dir=cache_dir)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name_or_path,
                                                           do_lower_case=do_lower_case,
                                                           cache_dir=cache_dir)
    model = models.BertForWikiQA.from_pretrained(model_name_or_path,
                                                 from_tf=bool('.ckpt' in model_name_or_path),
                                                 config=config,
                                                 cache_dir=cache_dir)

    ####### LOAD DATA
    dataset = DatasetManager(
        train_file, valid_file, test_file,
        max_seq_length,
        tokenizer,
        pad_position='right',
        device=device,
        logger=logger,
        mask_padding_with_zero=True,
        pad_token_segment_id=0,
        pad_token=0
    )

    # move model to CUDA
    model = model.cuda(device=device)

    # define optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    logger.info("Selected model has {} weight, {} of which has to be trained".format(
        sum([np.prod(p.size()) for p in optimizer_grouped_parameters[0]['params']]),
        sum([np.prod(p.size()) for p in optimizer_grouped_parameters[1]['params']]),
    ))

    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=20)
    
    # If using fp16
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    ####### TRAINING
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset.train))
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Instantaneous batch size per GPU = %d", batch_size)

    # Training
    set_seed(seed)

    def evaluate(eval_dataset, model, fp16=False):

        # Note that DistributedSampler samples randomly
        eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

        tot_right = 0
        tot = 0
        
        for step, batch in enumerate(eval_dataloader):
            model.eval()

            with torch.no_grad():
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                
                y_hat = model(**inputs)
                y_hat = torch.argmax(y_hat, dim=-1)
                tot_right += (y_hat == batch[3]).sum()
                tot += len(batch[3])

        logger.info("Accuracy is {}".format(float(tot_right) / tot))


    def train(train_dataset, eval_data, model, epochs_number, optimizer, scheduler, fp16=False):

        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    
        for epoch in range(epochs):
            logger.info("Training epoch {}".format(epoch+1))

            for step, batch in enumerate(train_dataloader):
                #print(tokenizer.decode(batch[0][0].cpu().numpy()))

                model.train()
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3]
                }

                outputs = model(**inputs)
                loss, results = outputs  # model outputs are always tuple in transformers (see doc)

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                #logger.info("Batch loss {}".format(loss.item()))

            logger.info("Evaluate")
            # evaluate
            evaluate(eval_data, model)

        
    train(dataset.train, dataset.valid, model, epochs, optimizer, scheduler, fp16=fp16)