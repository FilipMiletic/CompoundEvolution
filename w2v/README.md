# word2vec features

This directory contains the scripts to train word2vec models and then derive
time-specific features from them.

(1) `w2v_train.sh`
* Iterates over decade-level files (one-sentence-per-line corpus) and trains
  word2vec models using `w2v_train.py`.
* In the coarse-grained setting (30-year spans), this script moves 10-year
  corpus files into a temporary directory during training, and then moves them
  back out in the original location.

(2) `w2v_all_features.sh`
* Iterates over trained w2v models to derive feature information of four types.
* `w2v_synchronic_cos.py`: within-time features based on cosine scores
  (e.g., cosine for modifier-compound, head-compound, modifier-head, etc.)
* `w2v_synchronic_neighb.py`: within-time features based on nearest neighbors
  for a given pair of targets (e.g., compound and modifier):
  * overlap of two lists of neighbors;
  * cosine taken over the vectors of target-neighbor cosine scores.
* `w2v_diachronic_cos.py`: across-time features based on cosine scores
  (e.g., cosine for modifier in t1 and t2, etc.)
* `w2v_diachronic_neighb.py`: across-time features based on nearest neighbors
  for a given pair of targets (e.g., modifier in t1 and t2), as above.
