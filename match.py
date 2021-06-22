import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree
from feature_space import *

def BruteForce_init(featureSpace_A):
     nbrs = NearestNeighbors(algorithm='brute', n_neighbors=1).fit(featureSpace_A.T)
     return nbrs
    
def BallTree_init(featureSpace_A):
    tree = BallTree(featureSpace_A.T) 
    return tree

def MatcherInit(featureSpace_A, matcher_type):
    if matcher_type == 'BallTree':
        obj = BallTree_init(featureSpace_A) 
    else:
        obj = BruteForce_init(featureSpace_A)
    return obj

def BruteForce_run(featureVec_B, nbrs):
    dist, result = nbrs.kneighbors(featureVec_B.T)
    return result

def BallTreeMatcher_run(featureVec_B, nbrs):
    dist, result = nbrs.query(featureVec_B.T, k=1)
    return result

def BestApproximateMatch(fb, matcher_type, matcher_obj, wb):
    fbr = fb.reshape([-1, 1])
    if matcher_type == 'BallTree':
        ind_a = BallTreeMatcher_run(fbr, matcher_obj)
    else:
        ind_a = BruteForce_run(fbr, matcher_obj)
    return (ind_a[0, 0] // wb, ind_a[0, 0] % wb)

def BestCoherenceMatch(Fal, fb, s, q, wa, ha, wb, hb):
    i, j = q
    
    nblist = np.array(
        [
            [i-2, j-2], 
            [i-2, j-1], 
            [i-2, j], 
            [i-2, j+1], 
            [i-2, j+2],
            [i-1, j-2], 
            [i-1, j-1], 
            [i-1, j], 
            [i-1, j+1], 
            [i, j+2],
            [i, j-2], 
            [i,j-1]
            ]
            )
    valid = np.ones((12,1))
   
    for cur in range(nblist.shape[0]):
        outofBound = (nblist[cur, 0] < 0) or (nblist[cur, 0] >= hb) or (nblist[cur, 1] < 0) or (nblist[cur, 1] >= wb)
        if outofBound:
            valid[cur] = 0
            continue
        sr_cur = s[nblist[cur, 0], nblist[cur,1 ], :]
        pc_candidate = sr_cur + q - nblist[cur]
        outofBoundAfterOffset = (pc_candidate[0] < 0) or (pc_candidate[0] >= ha) or (pc_candidate[1] < 0) or (pc_candidate[1] >= wa)
        if outofBoundAfterOffset:
            valid[cur] = 0
    
    # nblist is the valid neighbours
    nblist = nblist[valid[:, 0] == 1, :]
    nb_no = nblist.shape[0]
    if nb_no == 0:
        return (-1, -1)
    
    # mapping_x,mapping_y is the location of mapped points in (A, A') pyramid
    sr = s[nblist[:,0], nblist[:,1], :]
    feature_ind = np.zeros((nb_no, 2), dtype='int32')
    feature_dim = Fal.shape[0]
    candidatates = np.zeros((feature_dim, nb_no))
    for nb in range(nb_no):
        feature_ind[nb] = sr[nb, :] + q - nblist[nb]
        candidatates[:, nb] = Fal[:, feature_ind[nb][0] * wa + feature_ind[nb][1]]
    dists = np.square(candidatates - fb.reshape([-1, 1]))
    nb_ind = np.argmin(np.sum(dists, keepdims=True), axis=1)
    return feature_ind[nb_ind[0], :]
    
def BestMatch(Fa, Fb, Ap, Bp, s, q, ANN, ANNType, l, K):
    Fbq = feature_vector_q(Fb[l], Bp, q, l)
    
    # If level = 0 or q is on the image border, not all the feature is valid, then Brute Force is used to find the NN
    mask = np.isnan(Fbq)
    ha, wa = Ap[l].shape
    hb, wb = Ap[l].shape
    L = len(Bp) - 1

    coarseWeight = createCoarseWeight()
    fineWeight = createFineWeight()
    valid = np.invert(mask)

    if (mask == False).all():
        p_a = BestApproximateMatch(Fbq, ANNType, ANN[l], wb)
        p_c = BestCoherenceMatch(Fa[l], Fbq, s[l], q, wa, ha, wb, hb)
    else:
        fbv = Fbq[valid]
        Fav = Fa[l][valid, :]
        BF = MatcherInit(Fav, 'BF')
        p_a = BestApproximateMatch(fbv, 'BF', BF, wb)
        p_c = BestCoherenceMatch(Fav, fbv, s[l], q, wa, ha, wb, hb)

    # If no result returned from BestCoherenceMatch, only return the result of BestApproximateMatch 
    if p_c[0] == -1:
        p = p_a
        return p

    # Compare the results of BestApproximateMatch and BestCoherenceMatch
    Fap_app = Fa[l][:, p_a[0] * wa + p_a[1]].reshape([-1, 1])
    Fap_app_normalize = np.vstack(
        [
            feature_vector_normalize(Fap_app[0:34], coarseWeight, fineWeight, False),
            feature_vector_normalize(Fap_app[34::], coarseWeight, fineWeight, True)
            ]
            )
    Fap_app_normalize = Fap_app_normalize[valid, :]

    Fap_coh = Fa[l][:, p_c[0] * wa + p_c[1]].reshape([-1, 1])
    Fap_coh_normalize = np.vstack(
        [
            feature_vector_normalize(Fap_coh[0:34], coarseWeight, fineWeight, False),
            feature_vector_normalize(Fap_coh[34::], coarseWeight, fineWeight, True)
            ]
            )
    Fap_coh_normalize = Fap_coh_normalize[valid, :]

    Fbq = Fbq.reshape([-1, 1])
    Fbq_normalize = np.vstack(
        [
            feature_vector_normalize(Fbq[0:34], coarseWeight, fineWeight, False),
            feature_vector_normalize(Fbq[34::], coarseWeight, fineWeight, True)
            ]
            )
    Fbq_normalize = Fbq_normalize[valid, :]

    d_a = (np.sum(np.square(Fap_app_normalize - Fbq_normalize)))
    d_c = (np.sum(np.square(Fap_coh_normalize - Fbq_normalize)))
    if d_c / d_a <= (1 + 2 ** (l-L) * K):
        p = p_c
    else:
        p = p_a
    return p