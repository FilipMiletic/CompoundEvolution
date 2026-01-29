import numpy as np
from scipy.spatial.distance import cosine

def get_safe_vec(target, model):

    try:
        vec = model.wv[target]
    except KeyError:
        vec = np.nan
        
    return vec


def get_safe_mean(list_of_vecs):
    
    if any(vec is np.nan for vec in list_of_vecs):
        mean = np.nan
    else:
        mean = np.mean(list_of_vecs, axis=0)
    
    return mean


def get_safe_cos(vec1, vec2, distance=False):
    
    if vec1 is np.nan or vec2 is np.nan:
        cos = np.nan
    else:
        cos = cosine(vec1, vec2)
        if not distance:
            cos = 1 - cos
    
    return cos


def get_safe_nns(target, model, k=10):
    
    try:
        nns = model.wv.most_similar(target, topn=k)
    except KeyError:
        nns = np.nan
        
    return nns


def get_nns_pooledvec(targets, model, k=10):
    
    vecs = []
    for target in targets:
        vec = get_safe_vec(target, model)
        vecs.append(vec)
            
    custom_vec = get_safe_mean(vecs)
    
    if custom_vec is np.nan:
        nns = np.nan
    else:
        nns = model.wv.similar_by_vector(custom_vec, topn=k)
    
    return nns


def get_nns_overlap(nns1, nns2, distance=False):
    '''Expects lists of NNs output by gensim w2v'''
    
    if nns1 is np.nan or nns2 is np.nan:
        score = np.nan
    else:
        assert len(nns1) == len(nns2)
        k = len(nns1)
        nns1 = {nn for nn, _ in nns1}
        nns2 = {nn for nn, _ in nns2}
        overlap = nns1 & nns2
        
        score = 1 - len(overlap)/k
        if not distance:
            score = 1 - score
        
    return score


def get_nns_union(nns1, nns2):
    
    if nns1 is np.nan or nns2 is np.nan:
        union = np.nan
    else:
        assert len(nns1) == len(nns2)
        nns1 = {nn for nn, _ in nns1}
        nns2 = {nn for nn, _ in nns2}
        union = nns1 | nns2
    
    return union


def get_secondorder_cos(target, nns, model1, model2, distance=False):
    
    if nns is np.nan:
        cos = np.nan
    else:     
        sims1 = []
        sims2 = []

        if isinstance(target, list): # mh vector
            target1_vecs = [get_safe_vec(t, model1) for t in target]
            target2_vecs = [get_safe_vec(t, model2) for t in target]
            target1 = get_safe_mean(target1_vecs)
            target2 = get_safe_mean(target2_vecs)            
        else:
            target1 = get_safe_vec(target, model1)
            target2 = get_safe_vec(target, model2)

        for nn in nns:
            vec1 = get_safe_vec(nn, model1)
            vec2 = get_safe_vec(nn, model2)

            cos1 = get_safe_cos(target1, vec1)
            cos2 = get_safe_cos(target2, vec2)

            if cos1 is not np.nan and cos2 is not np.nan:
                sims1.append(cos1)
                sims2.append(cos2)

        if not sims1:
            cos = np.nan
        else:
            cos = cosine(sims1, sims2)
            if not distance:
                cos = 1 - cos
    
    return cos


def get_secondorder_cos_syn(target1, target2, nns, model, distance=False):
    """Synchronic version."""
    
    if nns is np.nan:
        cos = np.nan
    else:     
        sims1 = []
        sims2 = []

        target1 = get_safe_vec(target1, model)
        
        if isinstance(target2, list): # mh vector
            target2_vecs = [get_safe_vec(t, model) for t in target2]
            target2 = get_safe_mean(target2_vecs)
        else:
            target2 = get_safe_vec(target2, model)

        for nn in nns:
            vec = get_safe_vec(nn, model)

            cos1 = get_safe_cos(target1, vec)
            cos2 = get_safe_cos(target2, vec)

            if cos1 is not np.nan and cos2 is not np.nan:
                sims1.append(cos1)
                sims2.append(cos2)

        if not sims1:
            cos = np.nan
        else:
            cos = cosine(sims1, sims2)
            if not distance:
                cos = 1 - cos
    
    return cos