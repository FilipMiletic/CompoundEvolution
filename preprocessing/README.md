# Corpus preprocessing

This directory contains the scripts to preprocess the data from CCOHA:
Clean Corpus of Historical American English (Alatrash et al., LREC 2020).

The original corpus is in a CoNLL-like format, with one zip file per decade.  
The scripts should be run in the order described below.

(1) `process_ccoha.sh`
* Runs `process_ccoha.py` on each decade-level zip file.
* Based on an input list of target compounds, processes their occurrences
  (hyphenated, space-separated) into underscore-joined versions.
* Removes punctuation and noise-like parts-of-speech.
* Outputs corpus in one-sentence-per-line format, with each token represented as
  `<lemma>::<pos>`.

(2) `prep_data_bert_comp.py`
* Processes the one-sentence-per-line corpus for BERT-like models.
* Based on a list of target compounds, makes compound-level corpus files.
* Removes underscores from compound lemmas, as well as POS tags.

(3) `prep_data_bert_const.py`
* Same as above, but for individual constituent words.