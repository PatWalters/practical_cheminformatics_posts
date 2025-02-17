#!/usr/bin/env python

from rdkit import Chem
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import osmordred as rd
import useful_rdkit_utils as uru
import numpy as np

def find_finite_descriptors(df,col):
    desc = df[col]
    good_cols = np.isfinite(np.stack(desc.values)).all(0)
    return good_cols

# Define descriptor computation function
def calc_osmordred(smiles, version=2):        
    if version == 1:
        " original version from Mordred"
        v = 1
        doExEstate = False
    else:
        " expended descriptors with more features and fixed InformationContent in cpp"
        v = 2
        doExEstate = True
        mol = Chem.MolFromSmiles(smiles)
        if smiles.find(".") >= 0:
            mol = uru.get_largest_fragment(mol)
    if mol is None:
        return None # Return an empty array instead of None
    results = []
    try:
        results.append(np.array(rd.CalcABCIndex(mol)))
        results.append(np.array(rd.CalcAcidBase(mol)))
        results.append(np.array(rd.CalcAdjacencyMatrix(mol, v))) # add sm1 removed in v1.1
        results.append(np.array(rd.CalcAromatic(mol)))
        results.append(np.array(rd.CalcAtomCount(mol, v)))  #  add nHetero in v1.1
        results.append(np.array(rd.CalcAutocorrelation(mol)))
        results.append(np.array(rd.CalcBCUT(mol)))
        results.append(np.array(rd.CalcBalabanJ(mol)))
        results.append(np.array(rd.CalcBaryszMatrix(mol)))
        results.append(np.array(rd.CalcBertzCT(mol)))
        results.append(np.array(rd.CalcBondCount(mol)))
        results.append(np.array(rd.CalcRNCGRPCG(mol))) # CPSA 2D descriptors on charges
        results.append(np.array(rd.CalcCarbonTypes(mol, v))) # add calcFractionCSP3 in v1.1
        results.append(np.array(rd.CalcChi(mol)))
        results.append(np.array(rd.CalcConstitutional(mol)))
        results.append(np.array(rd.CalcDetourMatrix(mol))) # add sm1 since v1.1
        results.append(np.array(rd.CalcDistanceMatrix(mol,v)))
        results.append(np.array(rd.CalcEState(mol, doExEstate))) # no impact True/False
        results.append(np.array(rd.CalcEccentricConnectivityIndex(mol)))
        results.append(np.array(rd.CalcExtendedTopochemicalAtom(mol)))
        results.append(np.array(rd.CalcFragmentComplexity(mol)))
        results.append(np.array(rd.CalcFramework(mol)))
        results.append(np.array(rd.CalcHydrogenBond(mol)))
        if version==1:
            results.append(CalcIC(mol))
        else:
            results.append(np.array(rd.CalcLogS(mol))) # added if version > 1!
            results.append(np.array(rd.CalcInformationContent(mol,5)))

        results.append(np.array(rd.CalcKappaShapeIndex(mol)))
        results.append(np.array(rd.CalcLipinski(mol)))
        results.append(np.array(rd.CalcMcGowanVolume(mol)))
        results.append(np.array(rd.CalcMoeType(mol)))
        results.append(np.array(rd.CalcMolecularDistanceEdge(mol)))
        results.append(np.array(rd.CalcMolecularId(mol)))
        results.append(np.array(rd.CalcPathCount(mol)))
        results.append(np.array(rd.CalcPolarizability(mol)))
        results.append(np.array(rd.CalcRingCount(mol)))
        results.append(np.array(rd.CalcRotatableBond(mol)))
        results.append(np.array(rd.CalcSLogP(mol)))
        results.append(np.array(rd.CalcTopoPSA(mol)))
        results.append(np.array(rd.CalcTopologicalCharge(mol)))
        results.append(np.array(rd.CalcTopologicalIndex(mol)))
        results.append(np.array(rd.CalcVdwVolumeABC(mol)))
        results.append(np.array(rd.CalcVertexAdjacencyInformation(mol)))
        results.append(np.array(rd.CalcWalkCount(mol)))
        results.append(np.array(rd.CalcWeight(mol)))
        results.append(np.array(rd.CalcWienerIndex(mol)))
        results.append(np.array(rd.CalcZagrebIndex(mol)))
        if version>1:
        #  new descriptors added
            results.append(np.array(rd.CalcPol(mol))) 
            results.append(np.array(rd.CalcMR(mol))) 
            results.append(np.array(rd.CalcODT(mol))) # not yet implemented return 1!
            results.append(np.array(rd.CalcFlexibility(mol))) 
            results.append(np.array(rd.CalcSchultz(mol)))
            results.append(np.array(rd.CalcAlphaKappaShapeIndex(mol))) 
            results.append(np.array(rd.CalcHEState(mol))) # very slightly slower
            results.append(np.array(rd.CalcBEState(mol))) # as a true impact
            results.append(np.array(rd.CalcAbrahams(mol))) # as a true impact : vf2 smartparser 
            # new triplet features x5 faster using combined Linear Equation resolution instead of per vector targets...
            results.append(np.array(rd.CalcANMat(mol)))
            results.append(np.array(rd.CalcASMat(mol)))
            results.append(np.array(rd.CalcAZMat(mol)))
            results.append(np.array(rd.CalcDSMat(mol)))
            results.append(np.array(rd.CalcDN2Mat(mol)))
            results.append(np.array(rd.CalcFrags(mol)))
            results.append(np.array(rd.CalcAddFeatures(mol)))
            
        results_to_concat = [np.atleast_1d(r) for r in results]
        return np.concatenate(results_to_concat)
    except Exception as e:
        print(f"Error processing molecule {smiles}: {e}")
        return None




def Calculate(smiles_list, n_jobs=4,  version=1):
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit tasks with their indices
        futures = {executor.submit(CalcOsmordred, smi, version): idx for idx, smi in enumerate(smiles_list)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing molecules"):
            idx = futures[future]  # Retrieve the index of the SMILES string
            try:
                result = future.result()
                if result is not None:
                    results.append((idx, result))  # Store the index and the result
            except Exception as e:
                print(f"Error processing molecule at index {idx}: {e}")

    # Sort results by the original index to maintain order
    results.sort(key=lambda x: x[0])
    ordered_results = [res[1] for res in results]
    return ordered_results





if __name__ == "__main__":
    print("Osmordred library contents:")
    print(dir(rd))
    version = 2
    smiles = ['CCCO','CCCN','c1ccccc1']
    smiles_list = smiles
    n_jobs = 1  # Number of cores to use

    print(f"Processing {len(smiles_list)} molecules with {n_jobs} cores...")
    results = Calculate(smiles_list, n_jobs=n_jobs, version=version)
    print(results)
    # Convert to DataFrame and save to file
    df_results = pd.DataFrame(results)
    
    print(f"Finished processing. Results shape: {df_results.shape}")
    df_results.to_csv('Myfeatures.csv', index=False)

    # additional wrappers from double/int into std::vector of double
    print(list(rd.CalcSchultz(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcPol(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcMR(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcODT(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcFlexibility(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcLogS(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcHydrogenBond(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcFramework(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcBertzCT(Chem.MolFromSmiles(smiles[-1]))))
    print(list(rd.CalcBalabanJ(Chem.MolFromSmiles(smiles[-1]))))

    print(list(rd.CalcInformationContent(Chem.MolFromSmiles(smiles[0]),5)))

