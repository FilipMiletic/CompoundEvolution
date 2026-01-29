import argparse
import logging
import os
import re
from gensim.models import word2vec


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', help='directory or single corpus files')
    parser.add_argument('model_path', help='directory where to save model')
    parser.add_argument('--dims', help='vector dimensions, default 100',
                        default=100, type=int)
    parser.add_argument('--win', help='window around target, default 5',
                        default=5, type=int)
    parser.add_argument('--freq', help='minimum frequency, default 5',
                        default=5, type=int)
    parser.add_argument('--algo', help="one of ['cbow', 'sg'], default 'sg'",
                        default='sg', choices=['cbow', 'sg'])
    parser.add_argument('--work', help='number of workers, default 24',
                        default=24, type=int)
    args = parser.parse_args()

    in_path = args.corpus_path
    out_path = args.model_path
    vector_size = args.dims
    window = args.win
    min_count = args.freq
    algo = args.algo
    sg = 1 if algo == 'sg' else 0
    workers = args.work

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    # Set up input data + years to use for model name
    if os.path.isdir(in_path):
        sentences = word2vec.PathLineSentences(in_path)
        years = [re.search('\d+', f).group() for f in os.listdir(in_path)]
        years = sorted(years)
        years = '-'.join([years[0], years[-1]])
    else:
        sentences = word2vec.LineSentence(in_path)
        years = os.path.basename(os.path.normpath(in_path))
        years = re.search('\d+', years).group()
    
    logging.info(f'Loading data from: {in_path}')
    logging.info(f"Training: dim {vector_size} - win {window} - "
                 f"freq {min_count} - algo {algo} - workers {workers}")
    
    # Train model
    model = word2vec.Word2Vec(sentences,
                              vector_size=vector_size,
                              window=window,
                              min_count=min_count,
                              sg=sg,
                              workers=workers)
    
    logging.info('Model trained.')
    
    # Save model
    #corpus = os.path.basename(os.path.normpath(in_path))
    #corpus = re.search('\d.+\d', corpus).group() # take just the years
    out_name = f'{years}_d{vector_size}-w{window}-f{min_count}-{algo}.model'
    out_path = os.path.join(out_path, out_name)
    model.save(out_path)
    
    logging.info(f'Model saved to {out_path}')
    
    
if __name__ == '__main__':
    main()