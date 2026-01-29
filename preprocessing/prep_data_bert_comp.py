"""Converts whole-corpus, lemmatized+tagged files (prepared for w2v) into
target-level, lemmatized files (for BERT)."""
import argparse
import os
import re
from collections import defaultdict
from smart_open import open

import sys
sys.path.append('../')
from utils.load_targets import load_targets

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_dir', help='one-sentence-per-line corpus')
    parser.add_argument('out_dir', help='where to write target-level files')
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir

    targets = set(load_targets())

    in_files = os.listdir(in_dir)
    in_files = [f for f in in_files if f.endswith('.txt.gz')]
    in_files = sorted(in_files)

    for in_file in in_files:

        year = re.search('\d+', in_file).group()
        target2line = defaultdict(list)

        with open(os.path.join(in_dir, in_file), 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = [t.split('::')[0] for t in line.split()]
            for target in set(line) & targets:
                target2line[target].append(line)

        write_out_dir = os.path.join(out_dir, year)
        if os.path.isdir(write_out_dir):
            raise ValueError('Output directory already exists.')

        os.mkdir(write_out_dir)

        for target in target2line:
            out_lines = []
            for line in target2line[target]:
                out_line = []
                for tok in line:
                    if tok == target:
                        tok = tok.replace('_', ' ')
                    out_line.append(tok)
                out_line = ' '.join(out_line) + '\n'
                out_lines.append(out_line)

            out_file = os.path.join(write_out_dir, target + '.txt.gz')
            with open(out_file, 'w') as f:
                f.writelines(out_lines)

if __name__ == '__main__':
    main()
