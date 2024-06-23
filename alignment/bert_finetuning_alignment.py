import argparse
import logging
import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F

from utils_alignment import load_align_corpus, filter_data_alignments, pick_aligned

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

from word_level_bert import WordLevelBert
from transformers import set_seed


def align_bert_multiple(train, model, model_base,
                        num_sentences, languages, batch_size,
                        save_path,
                        splitbatch_size=4, epochs=1,
                        learning_rate=0.00005, learning_rate_warmup_frac=0.1,
                        base_rate=1, include_clssep=True):
    # Adam hparams from Attention Is All You Need
    trainer = torch.optim.Adam([param for param in model.parameters() if
                                param.requires_grad], lr=1.,
                               betas=(0.9, 0.98), eps=1e-9)

    # set up functions to do linear lr warmup
    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr

    learning_rate_warmup_steps = int(num_sentences * learning_rate_warmup_frac)
    warmup_coeff = learning_rate / learning_rate_warmup_steps

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= learning_rate_warmup_steps:
            logging.info('Warming up, iter {}/{}'.format(iteration, learning_rate_warmup_steps))
            set_lr(iteration * warmup_coeff)

    model_base.eval()  # freeze and remember initial model

    total_processed = 0
    for epoch in tqdm.tqdm(range(epochs)):
        for i in range(0, num_sentences, batch_size):
            loss = None
            model.train()
            schedule_lr(total_processed // (len(languages)))
            for j, language in enumerate(languages):
                sent_1, sent_2, align = train[j]

                ii = i % len(sent_1)  # cyclic list - datasets may be diff sizes
                ss_1, ss_2 = sent_1[ii:ii + batch_size], sent_2[ii:ii + batch_size]

                aa = align[ii:ii + batch_size]

                # split batch to reduce memory usage
                for k in range(0, len(ss_1), splitbatch_size):
                    s_1 = ss_1[k:k + splitbatch_size]
                    s_2 = ss_2[k:k + splitbatch_size]

                    a = aa[k:k + splitbatch_size]

                    # pick out aligned indices in a packed representation
                    idx_1, idx_2 = pick_aligned(s_1, s_2, a, include_clssep)

                    # compute vectors for each position, pack the sentences
                    # result: packed_len x dim
                    ann_1, ann_2 = model(s_1), model(s_2)
                    ann_2_base = model_base(s_2)

                    loss_1 = F.mse_loss(ann_1[idx_1], ann_2[idx_2])
                    loss_2 = F.mse_loss(ann_2, ann_2_base)
                    loss_batch = loss_1 + base_rate * loss_2
                    if loss is None:
                        loss = loss_batch
                    else:
                        loss += loss_batch
                total_processed += len(ss_1)

            logging.info('Sentences {}-{}/{}, Loss: {}'.format(
                i, min(i + batch_size, num_sentences), num_sentences, loss))
            loss.backward()
            trainer.step()
            trainer.zero_grad()

    logging.info(f'Model is saved to {save_path}')
    model.bert.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser(description='Arguments for training.')
    
    parser.add_argument('--model_path', default='bert-base-multilingual-cased')
    parser.add_argument('--tokenizer_path', default='bert-base-multilingual-cased')

    parser.add_argument('--max_sent',
                        default=250000,
                        type=int,
                        action='store')
    parser.add_argument('--batch_size', default=32, type=int, action='store')
    parser.add_argument('--sent_path')
    parser.add_argument('--align_path')
    parser.add_argument('--save_path', default='./models/aligned_bert')
    parser.add_argument('--language', default='ru')
    parser.add_argument('--align_mode', choices=['end', 'start', 'avg', 'ori'])
    parser.add_argument('--base_rate', default=1, type=float, action='store')
    parser.add_argument('--learning_rate', default=0.00005, type=float, action='store')
    parser.add_argument('--seed', default=42, type=int, action='store')
    parser.add_argument('--include_clssep', action='store_true')
    args = parser.parse_args()


    model_path = args.model_path
    tokenizer_path = args.tokenizer_path

    max_sent = args.max_sent
    batch_size = args.batch_size
    sent_paths = [args.sent_path]
    align_paths = [args.align_path]
    languages = [args.language]
    align_mode = args.align_mode
    base_rate = args.base_rate
    learning_rate = args.learning_rate

    logging.info(f'Max sent: {max_sent}')
    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Sentences path: {args.sent_path}')
    logging.info(f'Alignment path: {args.align_path}')
    logging.info(f'Alignment mode: {align_mode}')
    logging.info(f'Rate of base model: {base_rate}')
    logging.info(f'Model will be saved to {args.save_path}')
    logging.info(f'Seed {args.seed}')
    logging.info(f'Learning rate {learning_rate}')
    logging.info(f'Include [CLS] and [SEP]: {args.include_clssep}')

    set_seed(args.seed)
    
    model = WordLevelBert(align_mode, model_path, tokenizer_path, batch_size=batch_size)
    model_base = WordLevelBert(align_mode, model_path, tokenizer_path, batch_size=batch_size)

    data = [load_align_corpus(sent_path, align_path, max_sent=max_sent) for
            sent_path, align_path in zip(sent_paths, align_paths)]

    filtered_data = filter_data_alignments(data, lang=args.language)

    num_sent = len(filtered_data[0][0])

    logging.info(f'Total sentences count: {num_sent}')

    align_bert_multiple(filtered_data, model, model_base, num_sent, languages, batch_size, args.save_path, 
                        learning_rate=learning_rate, base_rate=base_rate, include_clssep=args.include_clssep)


if __name__ == '__main__':
    main()
