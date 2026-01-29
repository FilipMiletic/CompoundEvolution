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
from orthogonal_procrustes import procrustes_align
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

    
    # Load models
    models = {}
    for grain in model_dirs:
        models[grain] = {}
        model_dir = model_dirs[grain]
        model_files = os.listdir(model_dir)
        model_files = [m for m in model_files if m.endswith('model')]
        model_files = sorted(model_files)

        for model_file in model_files:
            year = model_file.split('_')[0]
            models[grain][year] = Word2Vec.load(os.path.join(model_dir, model_file))
    
    logging.info('### Calculating diachronic cosines.')
    logging.info(f'Loaded models from {model_dir}.')
    master_scores = []

    end = os.path.basename(os.path.normpath(model_dir))
    if end.startswith('run'):
        run = f'-{end}'
    else:
        run = ''
        
    for grain in models:
        
        logging.info(f'Processing grain: {grain}.')
        
        years = list(models[grain].keys())
        year_pairs = list(zip(years[:-1], years[1:]))
        
        for year1, year2 in year_pairs:
            
            model1 = copy.deepcopy(models[grain][year1])
            model2 = copy.deepcopy(models[grain][year2])
            model2 = procrustes_align(model1, model2)

            for cpd, modif, head in zip(cpd_targets, modif_targets, head_targets):

                # Get vectors
                vec_c = [get_safe_vec(cpd, model1), get_safe_vec(cpd, model2)]
                vec_m = [get_safe_vec(modif, model1), get_safe_vec(modif, model2)]
                vec_h = [get_safe_vec(head, model1), get_safe_vec(head, model2)]
                vec_mh = [get_safe_mean([vec_m[0], vec_h[0]]),
                          get_safe_mean([vec_m[1], vec_h[1]])]

                # Compute cosines
                cos_c = get_safe_cos(*vec_c)
                cos_m = get_safe_cos(*vec_m)
                cos_h = get_safe_cos(*vec_h)
                cos_mh = get_safe_cos(*vec_mh)

                # Store cosines
                scores = {'cpd-time': cos_c,
                          'modif-time': cos_m,
                          'head-time': cos_h,
                          'const-time': cos_mh,
                         }

                cpd = cpd.replace('_', ' ')[:-4]
                for score_type in scores:
                    out_score = {'compound': cpd,
                                 'grain': grain,
                                 'year': '_'.join([year1, year2]),
                                 'measure': score_type+run,
                                 'score': scores[score_type]}
                    master_scores.append(out_score)

            logging.info(f'Processed {year1}-{year2}.')
        logging.info(f'Processed grain: {grain}.')
        
    scores_df = pd.DataFrame(master_scores)
    write_out_files(scores_df, out_dir)
    logging.info('Outputs written.')

    
if __name__ == '__main__':
    main()
