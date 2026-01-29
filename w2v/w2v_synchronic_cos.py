import argparse
import copy
import csv
import logging
import numpy as np
import os
import pandas as pd

logging.getLogger('gensim').setLevel(logging.WARNING)

from collections import defaultdict
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from w2v_feature_utils import *

import sys
sys.path.append('../')
from utils.load_targets import load_targets


def write_out_files(scores_df, out_dir):
    
    for grain in scores_df['grain'].unique():
        for measure in scores_df['measure'].unique():
            out_df = scores_df.loc[(scores_df['grain'] == grain) &
                                   (scores_df['measure'] == measure)]

            out_df = out_df.pivot(index='compound', columns='year', values='score')\
                           .reset_index()

            out_file = f'{grain}_w2v_{measure}.tsv'
            out_file = os.path.join(out_dir, out_file)

            if os.path.isfile(out_file):
                logging.warning(f'Not writing output because file exists: {out_file}')
            else:
                out_df.to_csv(out_file,
                              sep='\t',
                              index=False)

                
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('fine_dir', help='path to fine-grained w2v models')
    parser.add_argument('coarse_dir', help='path to coarse-grained w2v models')
    parser.add_argument('out_dir', help='output directory for features')

    args = parser.parse_args()  
    fine_dir = args.fine_dir
    coarse_dir = args.coarse_dir
    out_dir = args.out_dir
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                level=logging.INFO)  
        
    model_dirs = {'fine': fine_dir,
                  'coarse': coarse_dir}
    
    # Load targets + prepare them for w2v tags
    cpd_targets = load_targets(how='space')
    modif_targets = [t.split()[0]+'::nn' for t in cpd_targets]
    head_targets = [t.split()[1]+'::nn' for t in cpd_targets]
    cpd_targets = [t.replace(' ', '_')+'::nn' for t in cpd_targets]
    
    logging.info('### Calculating synchronic cosines.')
    master_scores = []

    for grain in model_dirs:
        
        model_dir = model_dirs[grain]
        model_files = os.listdir(model_dir)
        model_files = [m for m in model_files if m.endswith('model')]
        model_files = sorted(model_files)

        logging.info(f'Processing models from {model_dir}.')
        
        end = os.path.basename(os.path.normpath(model_dir))
        if end.startswith('run'):
            run = f'-{end}'
        else:
            run = ''
            
        for model_file in model_files:

            year = model_file.split('_')[0]
            model = Word2Vec.load(os.path.join(model_dir, model_file))

            for cpd, modif, head in zip(cpd_targets, modif_targets, head_targets):
                
                # Get vectors
                vec_c = get_safe_vec(cpd, model)
                vec_m = get_safe_vec(modif, model)
                vec_h = get_safe_vec(head, model)
                vec_mh = get_safe_mean([vec_m, vec_h])

                # Compute cosines
                cos_mh = get_safe_cos(vec_m, vec_h)
                cos_cm = get_safe_cos(vec_c, vec_m)
                cos_ch = get_safe_cos(vec_c, vec_h)
                cos_cmh = get_safe_cos(vec_c, vec_mh)
                cos_add = cos_cm + cos_ch
                cos_mult = cos_cm * cos_ch
                cos_comb = cos_add + cos_mult

                # Store cosines
                scores = {'cpd-modif': cos_cm,
                          'cpd-head': cos_ch,
                          'cpd-const': cos_cmh,
                          'cpd-add': cos_add,
                          'cpd-mult': cos_mult,
                          'cpd-comb': cos_comb,
                          'modif-head': cos_mh}

                cpd = cpd.replace('_', ' ')[:-4]
                for score_type in scores:
                    out_score = {'compound': cpd,
                                 'grain': grain,
                                 'year': year,
                                 'measure': score_type+run,
                                 'score': scores[score_type]}
                    master_scores.append(out_score)

            logging.info(f'Processed {year}.')
        logging.info(f'Processed grain: {grain}.')
        
    scores_df = pd.DataFrame(master_scores)
    write_out_files(scores_df, out_dir)
    logging.info('Outputs written.')

    
if __name__ == '__main__':
    main()
