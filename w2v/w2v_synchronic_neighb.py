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

    logging.info('### Calculating synchronic neighbors.')
    master_scores = []

    for grain in model_dirs:
        
        model_dir = model_dirs[grain]
        model_files = os.listdir(model_dir)
        model_files = [m for m in model_files if m.endswith('model')]
        model_files = sorted(model_files)

        logging.info(f'Processing k={k} for models from {model_dir}.')
        
        end = os.path.basename(os.path.normpath(model_dir))
        if end.startswith('run'):
            run = f'-{end}'
        else:
            run = ''
            
        for model_file in model_files:

            year = model_file.split('_')[0]
            model = Word2Vec.load(os.path.join(model_dir, model_file))

            for cpd, modif, head in zip(cpd_targets, modif_targets, head_targets):

                # Get neighbors
                nns_c = get_safe_nns(cpd, model, k)
                nns_m = get_safe_nns(modif, model, k)
                nns_h = get_safe_nns(head, model, k)
                nns_mh = get_nns_pooledvec([modif, head], model, k)

                # Get overlap measure (Béné/Gonen)
                overlap_mh = get_nns_overlap(nns_m, nns_h)
                overlap_cm = get_nns_overlap(nns_c, nns_m)
                overlap_ch = get_nns_overlap(nns_c, nns_h)
                overlap_cmh = get_nns_overlap(nns_c, nns_mh)
                overlap_add = overlap_cm + overlap_ch
                overlap_mult = overlap_cm * overlap_ch
                overlap_comb = overlap_add + overlap_mult

                # Get NNs unions (for second-order measure)
                union_mh = get_nns_union(nns_m, nns_h)
                union_cm = get_nns_union(nns_c, nns_m)
                union_ch = get_nns_union(nns_c, nns_h)
                union_cmh = get_nns_union(nns_c, nns_mh)

                # Get second-order measure (Hamilton)
                so_mh = get_secondorder_cos_syn(modif, head, union_mh, model)
                so_cm = get_secondorder_cos_syn(cpd, modif, union_cm, model)
                so_ch = get_secondorder_cos_syn(cpd, head, union_ch, model)
                so_cmh = get_secondorder_cos_syn(cpd, [modif, head], union_cmh, model)
                so_add = so_cm + so_ch
                so_mult = so_cm * so_ch
                so_comb = so_add + so_mult

                # Store scores
                scores = {'syn-nn-overlap-cm': overlap_cm,
                          'syn-nn-overlap-ch': overlap_ch,
                          'syn-nn-overlap-cmh': overlap_cmh,
                          'syn-nn-overlap-add': overlap_add,
                          'syn-nn-overlap-mult': overlap_mult,
                          'syn-nn-overlap-comb': overlap_comb,
                          'syn-nn-so-cm': so_cm,
                          'syn-nn-so-ch': so_ch,
                          'syn-nn-so-cmh': so_cmh,
                          'syn-nn-so-add': so_add,
                          'syn-nn-so-mult': so_mult,
                          'syn-nn-so-comb': so_comb,
                          'syn-nn-overlap-mh': overlap_mh,
                          'syn-nn-so-mh': so_mh}

                cpd = cpd.replace('_', ' ')[:-4]
                for score_type in scores:
                    out_score = {'compound': cpd,
                                 'grain': grain,
                                 'year': year,
                                 'measure': f'{score_type}-k{k}{run}',
                                 'score': scores[score_type]}
                    master_scores.append(out_score)

            logging.info(f'Processed {year}.')
        logging.info(f'Processed grain: {grain}.')
        
    scores_df = pd.DataFrame(master_scores)
    write_out_files(scores_df, out_dir)
    logging.info('Outputs written.')

    
if __name__ == '__main__':
    main()
