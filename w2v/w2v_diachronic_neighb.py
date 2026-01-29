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
    parser.add_argument('-k', help='number of neighbors, default: 10',
                         type=int, default=10)
    args = parser.parse_args()  
    fine_dir = args.fine_dir
    coarse_dir = args.coarse_dir
    out_dir = args.out_dir
    k = args.k
    
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

    logging.info('### Calculating diachronic neighbors.')        
    logging.info(f'Loaded models for k={k} from {model_dir}.')
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
            model1 = models[grain][year1]
            model2 = models[grain][year2]
        
            for cpd, modif, head in zip(cpd_targets, modif_targets, head_targets):

                # Get neighbors
                nns_c = [get_safe_nns(cpd, model1, k), 
                         get_safe_nns(cpd, model2, k)]
                nns_m = [get_safe_nns(modif, model1, k), 
                         get_safe_nns(modif, model2, k)]
                nns_h = [get_safe_nns(head, model1, k), 
                         get_safe_nns(head, model2, k)]
                nns_mh = [get_nns_pooledvec([modif, head], model1, k),
                          get_nns_pooledvec([modif, head], model2, k)]

                # Get overlap measure (Béné/Gonen)
                overlap_c = get_nns_overlap(*nns_c)
                overlap_m = get_nns_overlap(*nns_m)
                overlap_h = get_nns_overlap(*nns_h)
                overlap_mh = get_nns_overlap(*nns_mh)

                # Get NNs unions (for second-order measure)
                union_c = get_nns_union(*nns_c)
                union_m = get_nns_union(*nns_m)
                union_h = get_nns_union(*nns_h)
                union_mh = get_nns_union(*nns_mh)

                # Get second-order measure (Hamilton)
                so_params = [model1, model2]
                so_c = get_secondorder_cos(cpd, union_c, *so_params)
                so_m = get_secondorder_cos(modif, union_m, *so_params)
                so_h = get_secondorder_cos(head, union_h, *so_params)
                so_mh = get_secondorder_cos([modif, head], union_mh, *so_params)

                # Store scores
                scores = {'nn-overlap-cpd': overlap_c,
                          'nn-overlap-modif': overlap_m,
                          'nn-overlap-head': overlap_h,
                          'nn-overlap-const': overlap_mh,
                          'nn-so-cpd': so_c,
                          'nn-so-modif': so_m,
                          'nn-so-head': so_h,
                          'nn-so-const': so_mh
                         }

                cpd = cpd.replace('_', ' ')[:-4]
                for score_type in scores:
                    out_score = {'compound': cpd,
                                 'grain': grain,
                                 'year': '_'.join([year1, year2]),
                                 'measure': f'{score_type}-k{k}{run}',
                                 'score': scores[score_type]}
                    master_scores.append(out_score)

            logging.info(f'Processed {year1}-{year2}.')
        logging.info(f'Processed grain: {grain}.')
        
    scores_df = pd.DataFrame(master_scores)
    write_out_files(scores_df, out_dir)
    logging.info('Outputs written.')

    
if __name__ == '__main__':
    main()
