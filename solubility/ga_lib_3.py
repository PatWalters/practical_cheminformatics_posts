''' Library of code used to develop, validate, and run the SIMPD algorithm

Author: Greg Landrum (glandrum@ethz.ch)
'''
import sys
from rdkit import RDPaths

sys.path.append(RDPaths.RDContribDir)
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

from rdkit.Chem import Crippen
from SA_Score import sascorer

# The GA scenario targets
# scatter((74,80,72,78),(61,66,51,54),color='r')    

target_FG_vals = [(61,74),(66,80),(51,72),(54,78)]
delta_test_active_frac_vals = [.11, .30]


#*************************************************************************************
#***************************  This function runs the GA ******************************
#*************************************************************************************
from pymoo.core.problem import starmap_parallelized_eval
from pymoo.core.problem import elementwise_eval,looped_eval

from multiprocessing.pool import ThreadPool


def run_GA_old(df,
           strategy="CLUSTERS_SPLIT",
           pop_size=500,
           ngens=100,
           verbose=False,
           numThreads=1,
           seed_input=0xf00d,
           return_random_result=True,
           smilesCol='Smiles',
           actCol='Property_1',
           targetTrainFracActive=-1,
           targetTestFracActive=-1,
           targetDeltaTestFracActive=None,
           targetFval=None,
           targetGval=None,
           skipDescriptors=False):
    ''' Runs the GA using descriptors + activity distribution + F + G
    This is not the form used for SIMPD
    '''
    if numThreads > 1:
        pool = ThreadPool(numThreads)
        runner = pool.starmap
        func_eval = starmap_parallelized_eval
    else:
        runner = None
        func_eval = looped_eval

    random.seed(seed_input)
    np.random.seed(seed_input)

    sel_strategy = getattr(SelectionStrategy, strategy)
    df["mol"] = [
        Chem.MolFromSmiles(tmp_smi)
        for tmp_smi in df[smilesCol].to_numpy(dtype=str)
    ]

    # generate the descriptors we're using for the molecules:
    dvals = np.array([calc_descrs(m) for m in df.mol])
    dtgts = get_descr_targets()

    # generate the FPS we're using for the molecules:
    fps = get_fps(df.mol)

    # generate the distance matrix based on the fingerprints:
    dmat = np.zeros((len(fps), len(fps)), float)
    for i, fp in enumerate(fps):
        if i == len(fps) - 1:
            break
        ds = np.array(
            DataStructs.BulkTanimotoSimilarity(fp,
                                               fps[i + 1:],
                                               returnDistance=1))
        dmat[i, i + 1:] = ds
        dmat[i + 1:, i] = ds

    # generate the binned activity values
    #binned = [map_activity_to_idg_val((""), x*1000)[1] for x in df["Property_1"]];

    binned = df[actCol].to_numpy()
    binned[binned == "active"] = 1
    binned[binned == "inactive"] = 0
    binned = list(binned)

    if verbose:
        print('------------\nRunning GA')

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    max_pts = len(binned)
    #keep = 1000
    keep = max_pts
    n_max = int(0.2 * keep)

    if verbose:
        print(f'working with {keep} points and picking {n_max}')

    # do the clustering for the starting points
    distThreshold = 0.65
    clusterSizeThreshold = max(5, len(dmat) / 50)
    clusters = clusterData(dmat,
                           distThreshold,
                           clusterSizeThreshold=clusterSizeThreshold)

    if not skipDescriptors:
        problem = SplitProblem_NoR_TargetDescriptorDeltas(
            binned[:keep],
            fps[:keep],
            dmat[:keep, :keep],
            dvals[:keep],
            dtgts,
            n_max,
            runner=runner,func_eval=func_eval,
            clusters=clusters,
            targetTrainFracActive=targetTrainFracActive,
            targetTestFracActive=targetTestFracActive,
            targetDeltaTestFracActive=targetDeltaTestFracActive,
            targetFval=targetFval,targetGval=targetGval)
    else:
        problem = SplitProblem_JustFracActive(
            binned[:keep],
            fps[:keep],
            dmat[:keep, :keep],
            dvals[:keep],
            dtgts,
            n_max,
            runner=runner,func_eval=func_eval,
            clusters=clusters,
            targetTrainFracActive=targetTrainFracActive,
            targetTestFracActive=targetTestFracActive,
            targetDeltaTestFracActive=targetDeltaTestFracActive,
            targetFval=targetFval,targetGval=targetGval)

    algorithm = NSGA2(pop_size=pop_size,
                      sampling=ClusterSampling(selectionStrategy=sel_strategy,
                                               clusters=clusters),
                      crossover=BinaryCrossover2(),
                      mutation=MyMutation2(),
                      eliminate_duplicates=True)

    res = minimize(problem,
                   algorithm, ('n_gen', ngens),
                   seed=seed_input,
                   verbose=True)

    if verbose:
        print(f"{len(res.F)} solutions")
        print("Function value: %s" % res.F[0])

    #return res;

    #now bring the solutions in a nice format
    tests_inds = []
    train_inds = []

    for tmp_sol in range(len(res.F)):

        tests_inds.append(np.arange(len(res.X[tmp_sol]))[(res.X[tmp_sol])])
        train_inds.append(np.arange(len(res.X[tmp_sol]))[~(res.X[tmp_sol])])

    if return_random_result:
        sample_ind_returned = np.random.choice(np.arange(len(train_inds)))
        return train_inds[sample_ind_returned], tests_inds[
            sample_ind_returned], res
    else:
        return train_inds, tests_inds, res

def run_GA_SIMPD(df,
           strategy="CLUSTERS_SPLIT",
           pop_size=500,
           ngens=100,
           verbose=False,
           numThreads=1,
           seed_input=0xf00d,
           return_random_result=True,
           smilesCol='Smiles',
           actCol='Property_1',
           targetTrainFracActive=-1,
           targetTestFracActive=-1,
           targetDeltaTestFracActive=None):
    ''' Runs the GA using descriptors + activity distribution + (G-F) + G
    This is the form used for SIMPD
    '''

    if numThreads > 1:
        pool = ThreadPool(numThreads)
        runner = pool.starmap
        func_eval = starmap_parallelized_eval
    else:
        runner = None
        func_eval = looped_eval

    random.seed(seed_input)
    np.random.seed(seed_input)

    sel_strategy = getattr(SelectionStrategy, strategy)
    df["mol"] = [
        Chem.MolFromSmiles(tmp_smi)
        for tmp_smi in df[smilesCol].to_numpy(dtype=str)
    ]

    # generate the descriptors we're using for the molecules:
    dvals = np.array([calc_descrs(m) for m in df.mol])
    dtgts = get_descr_targets()

    # generate the FPS we're using for the molecules:
    fps = get_fps(df.mol)

    # generate the distance matrix based on the fingerprints:
    dmat = np.zeros((len(fps), len(fps)), float)
    for i, fp in enumerate(fps):
        if i == len(fps) - 1:
            break
        ds = np.array(
            DataStructs.BulkTanimotoSimilarity(fp,
                                               fps[i + 1:],
                                               returnDistance=1))
        dmat[i, i + 1:] = ds
        dmat[i + 1:, i] = ds

    # generate the binned activity values
    #binned = [map_activity_to_idg_val((""), x*1000)[1] for x in df["Property_1"]];

    binned = df[actCol].to_numpy()
    binned[binned == "active"] = 1
    binned[binned == "inactive"] = 0
    binned = list(binned)

    if verbose:
        print('------------\nRunning GA')

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    max_pts = len(binned)
    #keep = 1000
    keep = max_pts
    n_max = int(0.2 * keep)

    if verbose:
        print(f'working with {keep} points and picking {n_max}')

    # do the clustering for the starting points
    distThreshold = 0.65
#    clusterSizeThreshold = max(5, len(dmat) / 50)
    clusterSizeThreshold = 5
    clusters = clusterData(dmat,
                           distThreshold,
                           clusterSizeThreshold=clusterSizeThreshold)

    problem = SplitProblem_DescriptorAndFGDeltas(
        binned[:keep],
        fps[:keep],
        dmat[:keep, :keep],
        dvals[:keep],
        dtgts,
        n_max,
        runner=runner,func_eval=func_eval,
        clusters=clusters,
        targetTrainFracActive=targetTrainFracActive,
        targetTestFracActive=targetTestFracActive,
        targetDeltaTestFracActive=targetDeltaTestFracActive)


    algorithm = NSGA2(pop_size=pop_size,
                      sampling=ClusterSampling(selectionStrategy=sel_strategy,
                                               clusters=clusters),
                      crossover=BinaryCrossover2(),
                      mutation=MyMutation2(),
                      eliminate_duplicates=True)

    res = minimize(problem,
                   algorithm, ('n_gen', ngens),
                   seed=seed_input,
                   verbose=True)

    if verbose:
        print(f"{len(res.F)} solutions")
        print("Function value: %s" % res.F[0])

    #return res;

    #now bring the solutions in a nice format
    tests_inds = []
    train_inds = []

    for tmp_sol in range(len(res.F)):

        tests_inds.append(np.arange(len(res.X[tmp_sol]))[(res.X[tmp_sol])])
        train_inds.append(np.arange(len(res.X[tmp_sol]))[~(res.X[tmp_sol])])

    if return_random_result:
        sample_ind_returned = np.random.choice(np.arange(len(train_inds)))
        return train_inds[sample_ind_returned], tests_inds[
            sample_ind_returned], res
    else:
        return train_inds, tests_inds, res






#*************************************************************************************
#******************************  calculation of descriptors **************************
#*************************************************************************************

from collections import namedtuple

DescrTuple = namedtuple(
    'DescrTuple',
    ('descriptor', 'function', 'direction', 'target_fraction', 'target_value'))
descrs = [
    DescrTuple('SA_Score', sascorer.calculateScore, 1, 0.10, 0.10 * 2.8),
    DescrTuple('HeavyAtomCount', lambda x: x.GetNumHeavyAtoms(), 1, 0.1,
               0.1 * 31),
    DescrTuple('TPSA', Descriptors.TPSA, 1, 0.15, 0.15 * 88.),
    DescrTuple(
        'fr_benzene/1000 HeavyAtoms',
        lambda x: 1000 * Descriptors.fr_benzene(x) / x.GetNumHeavyAtoms(), -1,
        -0.2, -0.2 * 44),
]


def calc_descrs(mol):
    res = []
    for itm in descrs:
        res.append(itm.function(mol))
    return res


def get_descr_targets():
    res = []
    for itm in descrs:
        res.append(itm.target_value)
    return res


#************************************************************************************
#******************************* GA utilities ***************************************
#************************************************************************************

import numpy as np
import random
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import bisect


def get_fps(ms, generator=None):
    " generate fingerprints for a set of molecules "
    if generator is None:
        #generator = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=6)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    fps = [generator.GetFingerprint(x) for x in ms]
    return fps


def get_distance_cdf(fps1,
                     fps2,
                     removeSelf=False,
                     vals=np.arange(0, 1.01, 0.01)):
    ' finds the CDF for the closest member of fps2 to each element of fps1 '
    nbrds = []
    for i, fp in enumerate(fps1):
        ds = DataStructs.BulkTanimotoSimilarity(fp, fps2, returnDistance=1)
        if removeSelf:
            ds.pop(i)
        nbrds.append(min(ds))
    nPts = len(nbrds)
    nbrds = np.array(nbrds)
    res = []
    for v in vals:
        res.append(np.sum(nbrds <= v) / nPts)
    return res


def get_distance_cdf_dmat(dmat,
                          idx1,
                          idx2,
                          removeSelf=False,
                          vals=np.arange(0, 1.01, 0.01)):
    ''' CDF of number of points in fps2 which are less than distThresh 
      from each point in fps1 
    '''
    dmat = dmat[idx1][:, idx2]
    nPts = len(idx1)
    if removeSelf:
        for i in range(nPts):
            dmat[i, i] = 10
    dmat = np.min(dmat, axis=1)
    res = []
    for v in vals:
        res.append(np.sum(dmat <= v) / nPts)
    return res


def get_dissim_cdf(fps1,
                   fps2,
                   frac=0.2,
                   removeSelf=False,
                   vals=np.arange(0, 1.01, 0.01)):
    ' finds the CDF for the X percentile most-distant member of fps2 to each element of fps1 '
    nbrds = []
    refPos = int(frac * len(fps2))
    for i, fp in enumerate(fps1):
        ds = DataStructs.BulkTanimotoSimilarity(fp, fps2, returnDistance=1)
        if removeSelf:
            ds.pop(i)
        ds = sorted(ds)
        nbrds.append(ds[refPos])
    nPts = len(nbrds)
    nbrds = np.array(nbrds)
    res = []
    for v in vals:
        res.append(np.sum(nbrds <= v) / nPts)
    return res


def get_randomdist_cdf(fps1,
                       fps2,
                       distThresh=0.8,
                       removeSelf=False,
                       vals=np.arange(0, 1.01, 0.01)):
    ''' CDF of number of points in fps2 which are greater than distThresh 
      from each point in fps1 
    '''
    nbrds = []
    nfps2 = len(fps2)
    for i, fp in enumerate(fps1):
        ds = DataStructs.BulkTanimotoSimilarity(fp, fps2, returnDistance=1)
        if removeSelf:
            ds.pop(i)
        ds = sorted(ds)
        pos = bisect.bisect_left(ds, distThresh)
        nbrds.append((nfps2 - pos) / nfps2)
        nPts = len(nbrds)
    nbrds = np.array(nbrds)
    res = []
    for v in vals:
        res.append(np.sum(nbrds <= v) / nPts)
    return res


def get_randomdist_cdf_dmat(dmat,
                            idx1,
                            idx2,
                            distThresh=0.8,
                            removeSelf=False,
                            vals=np.arange(0, 1.01, 0.01)):
    ''' CDF of number of points in fps2 which are greater than distThresh 
      from each point in fps1 
    '''
    dmat = dmat[idx1][:, idx2]
    nbrds = []
    nfps2 = dmat.shape[1]
    for i in range(dmat.shape[0]):
        pos = sum(dmat[i] < distThresh)
        if removeSelf:
            # the zero distance is always at the beginning
            pos += 1

        nbrds.append((nfps2 - pos) / nfps2)
    nPts = dmat.shape[0]
    if removeSelf:
        nPts -= 1
    nbrds = np.array(nbrds)
    res = []
    for v in vals:
        res.append(np.sum(nbrds <= v) / nPts)
    return res


def get_relateddist_cdf(fps1,
                        fps2,
                        distThresh=0.65,
                        removeSelf=False,
                        vals=np.arange(0, 1.01, 0.01)):
    ''' CDF of number of points in fps2 which are less than distThresh 
      from each point in fps1 
    '''
    nbrds = []
    nfps2 = len(fps2)
    for i, fp in enumerate(fps1):
        ds = DataStructs.BulkTanimotoSimilarity(fp, fps2, returnDistance=1)
        if removeSelf:
            ds.pop(i)
        ds = sorted(ds)
        nbrds.append(bisect.bisect_left(ds, distThresh) / nfps2)
    nPts = len(nbrds)
    nbrds = np.array(nbrds)
    res = []
    for v in vals:
        res.append(np.sum(nbrds <= v) / nPts)

    return res


def get_relateddist_cdf_dmat(dmat,
                             idx1,
                             idx2,
                             distThresh=0.65,
                             removeSelf=False,
                             vals=np.arange(0, 1.01, 0.01)):
    ''' CDF of number of points in fps2 which are less than distThresh 
      from each point in fps1 
    '''
    dmat = dmat[idx1][:, idx2]
    nbrds = []
    nfps2 = dmat.shape[1]
    for i in range(dmat.shape[0]):
        pos = sum(dmat[i] < distThresh)
        if removeSelf:
            # the zero distance is always at the beginning
            pos -= 1
        nbrds.append(pos / nfps2)
    nPts = dmat.shape[0]
    nbrds = np.array(nbrds)
    res = []
    for v in vals:
        res.append(np.sum(nbrds <= v) / nPts)

    return res


def calc_spatial_stats(testfps,
                       trainfps,
                       vals=np.arange(0, 1.01, 0.01),
                       includeTestInBootstrap=True):
    g_vals = get_distance_cdf(testfps, testfps, removeSelf=True, vals=vals)
    tfps = trainfps
    if includeTestInBootstrap:
        tfps = tfps + testfps
    bootstrap = [
        tfps[x]
        for x in [random.randint(0,
                                 len(tfps) - 1) for x in range(len(tfps))]
    ]
    f_vals = get_distance_cdf(bootstrap, testfps, vals=vals)
    s_vals = [f - g for f, g in zip(f_vals, g_vals)]
    return vals, g_vals, f_vals, s_vals


def modified_spatial_stats(testfps,
                           trainfps,
                           vals=np.arange(0, 1.01, 0.01),
                           includeTestInBootstrap=True,
                           justTheBasics=False):
    ' calculates F using closest member of train to test instead of vice-versa '
    g_vals = get_distance_cdf(testfps, testfps, removeSelf=True, vals=vals)
    tfps = trainfps
    if includeTestInBootstrap:
        tfps = tfps + testfps
    bootstrap = [
        tfps[x]
        for x in [random.randint(0,
                                 len(tfps) - 1) for x in range(len(tfps))]
    ]
    f_vals = get_distance_cdf(testfps, bootstrap, vals=vals)
    s_vals = [f - g for f, g in zip(f_vals, g_vals)]
    if not justTheBasics:
        h_vals = get_dissim_cdf(testfps, bootstrap, vals=vals)
        r1_vals = get_randomdist_cdf(testfps, bootstrap, vals=vals)
        r2_vals = get_relateddist_cdf(testfps,
                                      testfps,
                                      vals=vals,
                                      removeSelf=True)
        r3_vals = get_relateddist_cdf(testfps, bootstrap, vals=vals)
        return vals, g_vals, f_vals, s_vals, h_vals, r1_vals, r2_vals, r3_vals
    else:
        return vals, g_vals, f_vals, s_vals


def modified_spatial_stats_dmat(dmat,
                                testIdx,
                                trainIdx,
                                vals=np.arange(0, 1.01, 0.01),
                                includeTestInBootstrap=True):
    ' calculates F using closest member of train to test instead of vice-versa '
    g_vals = get_distance_cdf_dmat(dmat,
                                   testIdx,
                                   testIdx,
                                   removeSelf=True,
                                   vals=vals)
    tidx = list(trainIdx)
    if includeTestInBootstrap:
        tidx += list(testIdx)

    bootstrap = [
        tidx[x]
        for x in [random.randint(0,
                                 len(tidx) - 1) for x in range(len(tidx))]
    ]
    bootstrap = np.array(bootstrap)
    f_vals = get_distance_cdf_dmat(dmat,
                                   testIdx,
                                   bootstrap,
                                   removeSelf=False,
                                   vals=vals)
    s_vals = [f - g for f, g in zip(f_vals, g_vals)]
    return vals, g_vals, f_vals, s_vals


import math


def map_activity_to_idg_val(target_desc, standard_activity):
    ''' target-class based assignment of active/inactive labels from https://link.springer.com/article/10.1186/s13321-018-0325-4'''
    if 'protein kinase' in target_desc:
        thresh = 7.5
        idg_class = "Protein Kinase"
    elif 'enzyme  protease' in target_desc:
        thresh = 7.0
        idg_class = "Protease"  # Note: this is an addition to the original scheme
    elif 'membrane receptor  7tm1' in target_desc:
        thresh = 7.0
        idg_class = "GPCR"
    elif 'nuclear receptor' in target_desc:
        thresh = 7.0
        idg_class = "Nuclear Receptor"
    elif 'ion channel' in target_desc:
        thresh = 6.5
        idg_class = "Ion Channel"
    else:
        thresh = 6.0
        idg_class = "Other"

    if standard_activity <= 0:
        return idg_class, 0
    pact = -1 * math.log10(standard_activity * 1e-9)
    if pact >= thresh:
        return idg_class, 1
    else:
        return idg_class, 0


def get_imbalanced_bins_orig(data,
                        tgt_frac=0.2,
                        step_size=0.5,
                        active_inactive_offset=0.5):
    vs = list(sorted(data, reverse=True))
    tgt = int(tgt_frac * len(vs))
    act = vs[tgt]
    # "round"
    binAct = int(act / step_size) * step_size
    while sum([1 for x in vs if x >= binAct]) < tgt:
        binAct -= step_size
    lowerAct = binAct - active_inactive_offset
    return binAct, lowerAct

def get_imbalanced_bins(data,
                        tgt_frac=0.2,
                        step_size=0.1,
                        active_inactive_offset=0.5,
                        tol = 0.01):
    vs = list(sorted(data, reverse=True))
    tgt = int(tgt_frac * len(vs))
    act = vs[tgt]
    # "round"
    binAct = int(act / step_size) * step_size
    remain = [x for x in vs if x >= binAct or x <= (binAct-active_inactive_offset)]
    fRemain = sum([1 for x in remain if x >= binAct]) / len(remain)
    d = abs(fRemain - tgt_frac)
    lastD = 1e8
    while d<lastD and d>tol:
        if fRemain < tgt_frac:
            binAct -= step_size
        else:
            binAct += step_size
        lastD = d
        remain = [x for x in vs if x >= binAct or x <= (binAct-active_inactive_offset)]
    lowerAct = binAct - active_inactive_offset
    return binAct, lowerAct

def score_pareto_solutions(Fs,weights):
    Fs = np.copy(Fs)
    qs = np.quantile(Fs,0.9,axis=0)
    maxv = np.max(np.abs(Fs),axis=0)
    for i,q in enumerate(qs):
        if q==0:
            qs[i] = maxv[i]
            if qs[i] == 0:
                qs[i] = 1
    Fs /= qs
    Fs = np.exp(Fs*-1)
    weights = np.array(weights,float)
    # normalize:
    weights /= np.sum(weights)
    
    Fs *= weights
    return np.sum(Fs,axis=1)


# ---------------------------------
#   Asymmetric Validation Embedding (AVE) implementation
# code adapted from the supplementary material from https://doi.org/10.1021/acs.jcim.7b00403
# ---------------------------------
def calc_AVE(actfps,
             inactfps,
             at_indices,
             av_indices,
             it_indices,
             iv_indices,
             calcVE=False,
             offsetDiagonal=False):
    # approach from https://doi.org/10.1007/978-3-030-50420-5_44
    activesTrainFPs = [actfps[x] for x in at_indices]
    activesTestFPs = [actfps[x] for x in av_indices]
    inactivesTrainFPs = [inactfps[x] for x in it_indices]
    inactivesTestFPs = [inactfps[x] for x in iv_indices]
    av_at_D = calcDistMat(activesTestFPs, activesTrainFPs)
    iv_at_D = calcDistMat(inactivesTestFPs, activesTrainFPs)
    av_it_D = calcDistMat(activesTestFPs, inactivesTrainFPs)
    iv_it_D = calcDistMat(inactivesTestFPs, inactivesTrainFPs)

    if offsetDiagonal:
        av_at_D += np.identity(len(at_indices))
        iv_it_D += np.identity(len(it_indices))

    av_it_d = np.min(av_it_D, axis=1)
    av_at_d = np.min(av_at_D, axis=1)
    iv_at_d = np.min(iv_at_D, axis=1)
    iv_it_d = np.min(iv_it_D, axis=1)

    av_term = np.mean(av_it_d - av_at_d)
    iv_term = np.mean(iv_at_d - iv_it_d)
    if not calcVE:
        return av_term + iv_term
    else:
        return np.sqrt(av_term * av_term + iv_term * iv_term)


def calc_AVE_from_dists(actives_D,
                        inactives_D,
                        actives_inactives_D,
                        at_indices,
                        av_indices,
                        it_indices,
                        iv_indices,
                        calcVE=False,
                        offsetDiagonal=False):
    # approach from https://doi.org/10.1007/978-3-030-50420-5_44
    av_it_D = actives_inactives_D[av_indices, :][:, it_indices]
    av_at_D = actives_D[av_indices, :][:, at_indices]
    iv_at_D = actives_inactives_D.transpose()[iv_indices, :][:, at_indices]
    iv_it_D = inactives_D[iv_indices, :][:, it_indices]
    if offsetDiagonal:
        av_at_D += np.identity(len(at_indices))
        iv_it_D += np.identity(len(it_indices))
    av_term = np.mean(np.min(av_it_D, axis=1) - np.min(av_at_D, axis=1))
    iv_term = np.mean(np.min(iv_at_D, axis=1) - np.min(iv_it_D, axis=1))
    if not calcVE:
        return av_term + iv_term
    else:
        return np.sqrt(av_term * av_term + iv_term * iv_term)


from rdkit.ML.InfoTheory import rdInfoTheory


def population_cluster_entropy(X, clusters):
    if len(clusters) <= 1:
        return 0
    ccounts = np.zeros(len(clusters), int)
    for i, clust in enumerate(clusters):
        for entry in clust:
            if X[entry]:
                ccounts[i] += 1
    return rdInfoTheory.InfoEntropy(ccounts) / rdInfoTheory.InfoEntropy(
        np.ones(len(clusters), int))


def population_tanimoto(pop1, pop2):
    denom = sum(pop1 | pop2)
    if not denom:
        return 0.0
    return sum(pop1 & pop2) / denom


#************************************************************************************
#******************************* GA utilities ***************************************
#************************************************************************************

from pymoo.core.problem import ElementwiseProblem,elementwise_eval,looped_eval

class SplitProblem_NoR_TargetDescriptorDeltas(ElementwiseProblem):
    ''' pymoo problem form not used in SIMPD '''
    def __init__(self,
                 binned_acts,
                 fps,
                 dmat,
                 dvals,
                 descriptor_delta_targets,
                 n_max,
                 clusters=None,
                 targetTrainFracActive=-1,
                 targetTestFracActive=-1,
                 targetDeltaTestFracActive=None,
                 targetFval=None,
                 targetGval=None,
                 **kwargs):
        assert len(binned_acts) == len(fps)
        assert len(fps) == len(dvals)
        self.acts = np.array(binned_acts)
        self.dvals = np.array(dvals)
        self.dtargets = np.array(descriptor_delta_targets)
        assert self.dvals.shape[1] == self.dtargets.shape[0]
        self.fps = np.zeros((len(fps), ), object)
        for i, fp in enumerate(fps):
            self.fps[i] = fp

        self.dmat = dmat
        self._nObjs = len(dvals[0])
        if targetTestFracActive > 0 or targetTrainFracActive > 0:
            self.tgtTestFrac = targetTestFracActive
            self.tgtTrainFrac = targetTrainFracActive
            self._nObjs += int(targetTestFracActive > 0) + int(
                targetTrainFracActive > 0)
            self.tgtFrac = None
            self.deltaTestFracActive = None
        elif targetDeltaTestFracActive is not None:
            self.deltaTestFracActive = targetDeltaTestFracActive
            self.tgtFrac = None
        else:
            self._nObjs += 1
            self.tgtFrac = binned_acts.count(1) / len(binned_acts)
            self.deltaTestFracActive = None
        self.tgtFval = targetFval
        if targetFval is not None:
            self._nObjs += 1
        self.tgtGval = targetGval
        if targetGval is not None:
            self._nObjs += 1
        super().__init__(n_var=len(dvals),
                         n_obj=self._nObjs,
                         n_constr=1,
                         **kwargs)
        self.n_max = n_max
        self.clusters = clusters

    def _evaluate(self, x, out, *args, **kwargs):
        train = np.median(self.dvals[~x], axis=0)
        test = np.median(self.dvals[x], axis=0)
        descr_deltas = test - train

        descr_objects = abs(descr_deltas - self.dtargets)
        objectives = list(descr_objects)

        train_acts = self.acts[~x]
        train_frac = np.sum(train_acts, axis=0) / len(train_acts)
        test_acts = self.acts[x]
        test_frac = np.sum(test_acts, axis=0) / len(test_acts)
        if self.tgtFrac is not None:
            objectives.append(abs(test_frac - self.tgtFrac))
        elif self.deltaTestFracActive is not None:
            dTestFracActive = test_frac - np.sum(self.acts, axis=0)/len(self.acts)
            # print(f'  {test_frac:.2f} {np.sum(self.acts, axis=0)/len(self.acts):.2f} {dTestFracActive:.2f}')
            objectives.append(abs(self.deltaTestFracActive - dTestFracActive))
        else:
            if self.tgtTrainFrac > 0:
                objectives.append(abs(train_frac - self.tgtTrainFrac))
            if self.tgtTestFrac > 0:
                objectives.append(abs(test_frac - self.tgtTestFrac))
        if self.tgtFval is not None or self.tgtGval is not None:
            allIdx = np.arange(0, len(x), dtype=int)
            testIdx = allIdx[x]
            trainIdx = allIdx[~x]
            vals, g_vals, f_vals, s_vals = modified_spatial_stats_dmat(
                self.dmat, testIdx, trainIdx, includeTestInBootstrap=False)
            if self.tgtFval is not None:
                sum_F = np.sum(f_vals)
                objectives.append(abs(sum_F - self.tgtFval))
            if self.tgtGval is not None:
                sum_G = np.sum(g_vals)
                objectives.append(abs(sum_G - self.tgtGval))
            
        # objectives:
        out["F"] = objectives
        # constraints:
        out["G"] = [(self.n_max - np.sum(x))**2]
        if self.clusters:
            # keep the entropy below 0.9
            out["G"].append(population_cluster_entropy(x, self.clusters) - 0.9)


class SplitProblem_JustFracActive(ElementwiseProblem):
    ''' pymoo problem form not used in SIMPD '''
    def __init__(self,
                 binned_acts,
                 fps,
                 dmat,
                 dvals,
                 descriptor_delta_targets,
                 n_max,
                 clusters=None,
                 targetTrainFracActive=-1,
                 targetTestFracActive=-1,
                 targetDeltaTestFracActive=None,
                 targetFval=None,
                 targetGval=None,
                 **kwargs):
        assert len(binned_acts) == len(fps)
        assert len(fps) == len(dvals)
        self.acts = np.array(binned_acts)
        self.dmat = dmat

        self._nObjs = 0
        if targetTestFracActive > 0 or targetTrainFracActive > 0:
            self.tgtTestFrac = targetTestFracActive
            self.tgtTrainFrac = targetTrainFracActive
            self._nObjs += int(targetTestFracActive > 0) + int(
                targetTrainFracActive > 0)
            self.tgtFrac = None
            self.deltaTestFracActive = None
        elif targetDeltaTestFracActive is not None:
            self.deltaTestFracActive = targetDeltaTestFracActive
            self.tgtFrac = None
        else:
            self._nObjs += 1
            self.tgtFrac = binned_acts.count(1) / len(binned_acts)
            self.deltaTestFracActive = None
        super().__init__(n_var=len(dvals),
                         n_obj=self._nObjs,
                         n_constr=1,
                         **kwargs)
        self.n_max = n_max
        self.clusters = clusters

    def _evaluate(self, x, out, *args, **kwargs):
        objectives = []

        train_acts = self.acts[~x]
        train_frac = np.sum(train_acts, axis=0) / len(train_acts)
        test_acts = self.acts[x]
        test_frac = np.sum(test_acts, axis=0) / len(test_acts)
        if self.tgtFrac is not None:
            objectives.append(abs(test_frac - self.tgtFrac))
        elif self.deltaTestFracActive is not None:
            dTestFracActive = test_frac - np.sum(self.acts, axis=0)/len(self.acts)
            # print(f'  {test_frac:.2f} {np.sum(self.acts, axis=0)/len(self.acts):.2f} {dTestFracActive:.2f}')
            objectives.append(abs(self.deltaTestFracActive - dTestFracActive))
        else:
            if self.tgtTrainFrac > 0:
                objectives.append(abs(train_frac - self.tgtTrainFrac))
            if self.tgtTestFrac > 0:
                objectives.append(abs(test_frac - self.tgtTestFrac))
        # objectives:
        out["F"] = objectives
        # constraints:
        out["G"] = [(self.n_max - np.sum(x))**2]
        if self.clusters:
            # keep the entropy below 0.9
            out["G"].append(population_cluster_entropy(x, self.clusters) - 0.9)


class SplitProblem_DescriptorAndFGDeltas(ElementwiseProblem):
    ''' This is the pymoo Problem form used for SIMPD '''
    def __init__(self,
                 binned_acts,
                 fps,
                 dmat,
                 dvals,
                 descriptor_delta_targets,
                 n_max,
                 clusters=None,
                 targetTrainFracActive=-1,
                 targetTestFracActive=-1,
                 targetDeltaTestFracActive=None,
                 targetGFDeltaWindow=(10,30),
                 targetGval=70,
                 **kwargs):
        assert len(binned_acts) == len(fps)
        assert len(fps) == len(dvals)
        self.acts = np.array(binned_acts)
        self.dvals = np.array(dvals)
        self.dtargets = np.array(descriptor_delta_targets)
        assert self.dvals.shape[1] == self.dtargets.shape[0]
        self.fps = np.zeros((len(fps), ), object)
        for i, fp in enumerate(fps):
            self.fps[i] = fp

        self.dmat = dmat
        self._nObjs = len(dvals[0])
        # self._nObjs = 0
        if targetTestFracActive > 0 or targetTrainFracActive > 0:
            self.tgtTestFrac = targetTestFracActive
            self.tgtTrainFrac = targetTrainFracActive
            self._nObjs += int(targetTestFracActive > 0) + int(
                targetTrainFracActive > 0)
            self.tgtFrac = None
            self.deltaTestFracActive = None
        elif targetDeltaTestFracActive is not None:
            self.deltaTestFracActive = targetDeltaTestFracActive
            self.tgtFrac = None
        else:
            self._nObjs += 1
            self.tgtFrac = binned_acts.count(1) / len(binned_acts)
            self.deltaTestFracActive = None
        self.tgtDeltaGFWindow = targetGFDeltaWindow
        self._nObjs += 1
        self.tgtGval = targetGval
        self._nObjs += 1
        super().__init__(n_var=len(dvals),
                         n_obj=self._nObjs,
                         n_constr=1,
                         **kwargs)
        self.n_max = n_max
        self.clusters = clusters

    def _evaluate(self, x, out, *args, **kwargs):
        train = np.median(self.dvals[~x], axis=0)
        test = np.median(self.dvals[x], axis=0)
        descr_deltas = test - train

        descr_objects = abs(descr_deltas - self.dtargets)
        objectives = list(descr_objects)
        # objectives = []

        train_acts = self.acts[~x]
        train_frac = np.sum(train_acts, axis=0) / len(train_acts)
        test_acts = self.acts[x]
        test_frac = np.sum(test_acts, axis=0) / len(test_acts)
        if self.tgtFrac is not None:
            objectives.append(abs(test_frac - self.tgtFrac))
        elif self.deltaTestFracActive is not None:
            dTestFracActive = test_frac - np.sum(self.acts, axis=0)/len(self.acts)
            objectives.append(abs(self.deltaTestFracActive - dTestFracActive))
        else:
            if self.tgtTrainFrac > 0:
                objectives.append(abs(train_frac - self.tgtTrainFrac))
            if self.tgtTestFrac > 0:
                objectives.append(abs(test_frac - self.tgtTestFrac))
        allIdx = np.arange(0, len(x), dtype=int)
        testIdx = allIdx[x]
        trainIdx = allIdx[~x]
        vals, g_vals, f_vals, s_vals = modified_spatial_stats_dmat(
            self.dmat, testIdx, trainIdx, includeTestInBootstrap=False)
        sum_F = np.sum(f_vals)
        sum_G = np.sum(g_vals)
        delt = sum_G - sum_F
        if delt>self.tgtDeltaGFWindow[1]:
            objectives.append(delt-self.tgtDeltaGFWindow[1])
        elif delt<self.tgtDeltaGFWindow[0]:
            objectives.append(self.tgtDeltaGFWindow[0]-delt)
        else:
            objectives.append(0)

        if sum_G < self.tgtGval:
            objectives.append(self.tgtGval - sum_G)
        else:
            objectives.append(0)

        # objectives:
        out["F"] = objectives
        # constraints:
        out["G"] = [(self.n_max - np.sum(x))**2]
        if self.clusters:
            # keep the entropy below 0.9
            out["G"].append(population_cluster_entropy(x, self.clusters) - 0.9)



from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


class MySampling(Sampling):
    ''' simple sampling implementation for SIMPD
    
    This is adapted from the pymoo documentation
    '''
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X

class BinaryCrossover2(Crossover):
    ''' crossover implementation for SIMPD
    
    This is adapted from the pymoo documentation
    '''    
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation2(Mutation):
    ''' mutation implementation for SIMPD
    
    This is adapted from the pymoo documentation
    '''
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            # swap N% of the bits in each mutation
            swapFrac = 0.1
            X[i,
              np.random.choice(is_false, size=int(swapFrac *
                                                  problem.n_max))] = True
            X[i,
              np.random.choice(is_true, size=int(swapFrac *
                                                 problem.n_max))] = False

        return X


from rdkit.ML.Cluster import Butina
import random


class SelectionStrategy:
    DIVERSE = 1
    CLUSTERS_SPLIT = 2


def clusterData(dmat, threshold, clusterSizeThreshold, combineRandom=False):
    nfps = len(dmat)
    symmDmat = []
    # for i in range(1,len(tfps)):
    #     foo.extend(dmat3[i,:i])
    for i in range(1, nfps):
        symmDmat.extend(dmat[i, :i])
    cs = Butina.ClusterData(symmDmat,
                            nfps,
                            threshold,
                            isDistData=True,
                            reordering=True)
    cs = sorted(cs, key=lambda x: len(x), reverse=True)

    # start with the large clusters:
    largeClusters = [list(c) for c in cs if len(c) >= clusterSizeThreshold]
    if not largeClusters:
        raise ValueError("no clusters found")
    # print(f'{len(largeClusters)} large clusters found')
    # now combine the small clusters to make larger ones:
    if combineRandom:
        tmpCluster = []
        for c in cs:
            if len(c) >= clusterSizeThreshold:
                continue
            tmpCluster.extend(c)
            if len(tmpCluster) >= clusterSizeThreshold:
                random.shuffle(tmpCluster)
                largeClusters.append(tmpCluster)
                tmpCluster = []
        if tmpCluster:
            largeClusters.append(tmpCluster)
    else:
        # add points from small clusters to the nearest larger cluster
        #  nearest is defined by the nearest neighbor in that cluster
        oszs = [len(x) for x in largeClusters]
        for c in cs:
            if len(c) >= clusterSizeThreshold:
                continue
            for idx in c:
                closest = -1
                minD = 1e5
                for cidx, clust in enumerate(largeClusters):
                    for elem in clust:
                        d = dmat[idx, elem]
                        if d < minD:
                            closest = cidx
                            minD = d
                    #print('   ',idx,cidx,closest,minD)
                assert closest > -1
                largeClusters[closest].append(idx)
        # for cidx,clust in enumerate(largeClusters):
        #     print(cidx,oszs[cidx],len(clust),clust[-5:])
    return largeClusters


def assignUsingClusters(dmat,
                        threshold,
                        nTest,
                        clusterSizeThreshold=5,
                        nSamples=1,
                        selectionStrategy=SelectionStrategy.DIVERSE,
                        combineRandom=False,
                        clusters=None):
    if clusters is not None:
        largeClusters = clusters
    else:
        largeClusters = clusterData(dmat,
                                    threshold,
                                    clusterSizeThreshold,
                                    combineRandom=combineRandom)
    random.seed(0xf00d)
    res = []
    for i in range(nSamples):
        if selectionStrategy == SelectionStrategy.DIVERSE:
            ordered = []
            for c in largeClusters:
                # randomize the points in the cluster
                random.shuffle(c)
                ordered.extend((i / len(c), x) for i, x in enumerate(c))
            ordered = [y for x, y in sorted(ordered)]
            test = ordered[:nTest]
        else:
            random.shuffle(largeClusters)
            test = []
            for clus in largeClusters:
                nRequired = nTest - len(test)
                test.extend(clus[:nRequired])
                if len(test) >= nTest:
                    break
        res.append(test)
    return res


class ClusterSampling(Sampling):
    ''' cluster sampling implementation for SIMPD
    
    '''
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X

    def __init__(self,
                 distThreshold=0.65,
                 clusterSizeThreshold=-1,
                 selectionStrategy=SelectionStrategy.DIVERSE,
                 clusters=None):
        Sampling.__init__(self)
        self._distThreshold = distThreshold
        self._clusterSizeThreshold = clusterSizeThreshold
        self._selectionStrategy = selectionStrategy
        self._clusters = clusters

    def _do(self, problem, n_samples, **kwargs):
        if self._clusterSizeThreshold > 0:
            cst = self._clusterSizeThreshold
        else:
            cst = max(5, len(problem.dmat) / 50)

        assignments = assignUsingClusters(
            problem.dmat,
            self._distThreshold,
            problem.n_max,
            clusterSizeThreshold=cst,
            nSamples=n_samples,
            selectionStrategy=self._selectionStrategy,
            clusters=self._clusters)
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            I = assignments[k]
            X[k, I] = True

        return X

