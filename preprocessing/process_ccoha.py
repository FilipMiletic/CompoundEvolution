import argparse
import logging
import os
import pandas as pd
import re
import zipfile
from nltk.util import ngrams

from smart_open import open
from reduce_pos import reduce_tag


def get_targets(targets_file):
    """Loads target compounds to be joined in corpus data.
    
    Args:
        targets_file: TSV file where first column contains target compounds.
        
    Returns:
        A set of (modifier, head) tuples.
    """
    
    targets_df = pd.read_csv(targets_file, sep='\t')
    targets = targets_df.iloc[:, 0].to_list()
    targets = {tuple(t.split()) for t in targets}
    
    return targets


def process_sent(tokens, lemmas, tags, targets, keep_pos=False):
    """Processes a sentence read from CoNLL-like CCOHA format. 
    
    Args:
        tokens: List of tokens from target sentence.
        lemmas: Corresponding list of lemmas from target sentence.
        tags: Corresponding list of POS tags from target sentence.
        targets: Iterable of (modifier, head) tuples to join together.
        keep_pos: If true, appends POS tag to lemma in output.
        
    Returns:
        A string of whitespace-separated lemmas, without punctuation and
        <nul> tokens.
    """
    
    out = []
    out_tag = ''
    skip = False
    
    tag_excl = {'y',    # regular punctuation
                'q!',   # end-of-file tokens
                '"',    # quotation marks
                'z',    # -- symbol
                '...',  # sequences of periods
                'null', # varied minor stuff
                '!',    # exclamation point
                '-',    # dash
                }
    
    joint_targets = {'-'.join(t) for t in targets} |\
                    {''.join(t) for t in targets}
    
    for i, ngram in enumerate(ngrams(lemmas, 2, pad_right=True)):
        if ngram in targets:
            if tags[i] == tags[i+1] == 'nn':
                out_lemma = f'{lemmas[i]}_{lemmas[i+1]}'            
                if keep_pos:
                    #out_tag = f'::{tags[i]}_{tags[i+1]}'
                    out_tag = f'::{tags[i]}'
                out.append(out_lemma + out_tag)
                skip = True
        else:
            if (not skip
                and tags[i] not in tag_excl
                and lemmas[i] != '<nul>'):
                if (lemmas[i] in joint_targets 
                    and tags[i] == 'nn'):
                    out_lemma = lemmas[i].replace('-', '_')
                else:
                    out_lemma = lemmas[i]
                if keep_pos:
                    out_tag = f'::{tags[i]}'
                out.append(out_lemma + out_tag)
            skip = False
            
    return(' '.join(out))


def process_file(lines, targets, keep_pos=False):
    """Processes an individual CCOHA file.
    
    Args:
        lines: List of lines in the file.
        targets: Iterable of (modifier, head) tuples to join together.
        keep_pos: If true, appends POS tag to lemma in output.
        
    Returns:
        A list of processed lines, which contain whitespace-separated lemmas,
        optionally with POS tags, and without punctuation or <nul> lemmas.
    """
    
    tokens = []
    lemmas = []
    tags = []
    processed = []
    
    for i, line in enumerate(lines):
        if line.startswith('@@'):
            continue
        if line.startswith('<eos>'):
            out = process_sent(tokens, lemmas, tags, targets, 
                               keep_pos=keep_pos)
            processed.append(out)
            tokens = []
            lemmas = []
            tags = []
        else:
            line = line.rstrip().split('\t')
            token = line[0]
            lemma = line[1]
            tag = reduce_tag(line[2]).lower()
            tokens.append(token)
            lemmas.append(lemma)
            tags.append(tag)
            
    return processed


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('targets_file', help='file with compound targets, '
                        'expecting compounds in first tab-separated column')
    parser.add_argument('in_file', help='zipped CCOHA file to process')
    parser.add_argument('out_dir', help='directory where to write output')
    parser.add_argument('--fix_dir', help='directory with alternative file '
                        'versions, expecting format <decade>s_<filename>.txt')
    parser.add_argument('--pos', help='retain POS tags for targets/contexts',
                        action='store_true')
    
    args = parser.parse_args()
    targets_file = args.targets_file
    in_file = args.in_file
    out_dir = args.out_dir
    fix_dir = args.fix_dir
    keep_pos = args.pos
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    # Get set of (word1, word2) target tuples
    targets = get_targets(targets_file)
    logging.info(f'Loaded {len(targets)} compounds from: {targets_file}.')
    logging.info(f'Keeping POS tags for target/context lemmas: {keep_pos}.')
    
    # Set up alternative file versions (e.g. with fixed EOS tags)
    if fix_dir is not None:
        fix_files = os.listdir(fix_dir)
        logging.info(f'Using alternative corpus files from: {fix_dir}.')
    else:
        fix_files = []
    
    # Set up output file
    out_file = os.path.splitext(os.path.basename(in_file))[0] + '.txt.gz'
    out_file = os.path.join(out_dir, out_file)
    if os.path.isfile(out_file):
        raise ValueError('Output file already exists.')
    
    # Process input file
    with zipfile.ZipFile(in_file, mode='r') as zf:

        year = re.search('\d+', in_file).group()
        file_list = zf.namelist()
        
        logging.info(f'Reading {len(file_list)} files from {in_file}.')
        logging.info(f'Writing output to {out_file}.')
        
        for i, file in enumerate(file_list):
            
            fix_file = f'{year}s_{file}'
            if fix_file in fix_files:
                with open(os.path.join(fix_dir, fix_file), 'r') as f:
                    lines = f.readlines()
                    lines = [l.lower() for l in lines]
                logging.info(f'Reading from {fix_file}.')
            else:
                with zf.open(file) as f:
                    lines = f.readlines()
                    lines = [l.decode('utf-8').lower() for l in lines]
                    
            out_lines = process_file(lines, targets, keep_pos=keep_pos)
            with open(out_file, 'a') as of:
                of.writelines([l+'\n' for l in out_lines if l])
                    
            if (i+1) % 100 == 0:
                logging.info(f'Processed {i+1} files.')
    
    logging.info('Done.')


if __name__ == '__main__':
    main()