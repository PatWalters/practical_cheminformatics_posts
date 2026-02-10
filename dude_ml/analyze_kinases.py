import argparse
import logging
import sys
import time
from typing import Tuple, Optional, List, Dict

import pandas as pd
import requests
from Bio import Align
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_uniprot_sequence(gene_name: str, organism_id: str = "9606") -> Tuple[Optional[str], Optional[str]]:
    """
    Searches UniProt for a gene name and returns the ID and sequence.
    Defaults to Human (9606).
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    # Specific query: gene_exact match + organism + reviewed
    query = f"(gene_exact:{gene_name}) AND (organism_id:{organism_id}) AND (reviewed:true)"
    
    params = {
        "query": query,
        "format": "json",
        "fields": "accession,sequence",
        "size": 1
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            entry = data["results"][0]
            return entry["primaryAccession"], entry["sequence"]["value"]
        
        # Fallback 1: Relax "reviewed" constraint or gene_exact
        logger.debug(f"No reviewed entry found for {gene_name}, trying fallback search...")
        params["query"] = f"({gene_name}) AND (organism_id:{organism_id})"
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            entry = data["results"][0]
            return entry["primaryAccession"], entry["sequence"]["value"]
            
        logger.warning(f"No sequence found for {gene_name}")
        return None, None
            
    except requests.RequestException as e:
        logger.error(f"Network error fetching {gene_name}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error processing {gene_name}: {e}")
        return None, None

def calculate_similarity_matrix(results: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Calculates a pairwise sequence similarity matrix (percent identity).
    """
    logger.info("Calculating similarities...")
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    
    names = [r["Name"] for r in results]
    seq_data = {r["Name"]: r["Sequence"] for r in results}
    n = len(names)
    
    # Pre-fill structure
    matrix_data = {name: {other: 0.0 for other in names} for name in names}
    
    # Calculate upper triangle
    # Using tqdm for progress tracking
    total_ops = (n * (n + 1)) // 2
    pbar = tqdm(total=total_ops, desc="Aligning")
    
    for i in range(n):
        name1 = names[i]
        seq1 = seq_data[name1]
        
        # Diagonal is 100.0
        matrix_data[name1][name1] = 100.0
        pbar.update(1)
        
        for j in range(i + 1, n):
            name2 = names[j]
            seq2 = seq_data[name2]
            
            alignment = aligner.align(seq1, seq2)[0]
            
            # Biopython 1.86 compatible
            try:
                identities = alignment.counts().identities
            except AttributeError:
                 # Fallback for older versions if necessary, though requirements specify recent
                identities = 0
                for a, b in zip(*alignment):
                    if a == b:
                        identities += 1

            length = alignment.shape[1]
            
            perc_id = (identities / length * 100) if length > 0 else 0.0
            
            matrix_data[name1][name2] = perc_id
            matrix_data[name2][name1] = perc_id
            pbar.update(1)
            
    pbar.close()
    
    # Convert to DataFrame
    rows = []
    for name in names:
        row = matrix_data[name]
        row["Name"] = name
        rows.append(row)
        
    df_sim = pd.DataFrame(rows)
    # Reorder columns
    cols = ["Name"] + names
    df_sim = df_sim[cols]
    
    return df_sim

def main():
    parser = argparse.ArgumentParser(description="Fetch kinase sequences and calculate similarity.")
    parser.add_argument("--input", "-i", default="dud_kinase.txt", help="Input text file with kinase names")
    parser.add_argument("--output-seq", "-s", default="kinase_sequences.csv", help="Output CSV for sequences")
    parser.add_argument("--output-sim", "-m", default="kinase_similarities.csv", help="Output CSV for similarity matrix")
    args = parser.parse_args()

    # 1. Read kinase names
    try:
        with open(args.input, "r") as f:
            kinases = [line.strip() for line in f if line.strip()]
        logger.info(f"Found {len(kinases)} kinases in {args.input}.")
    except FileNotFoundError:
        logger.error(f"Input file {args.input} not found.")
        sys.exit(1)

    # 2. Fetch sequences
    results = []
    logger.info("Fetching sequences from UniProt...")
    
    for k in tqdm(kinases, desc="Fetching"):
        # Simple heuristic mapping for potential DUD specific names if standard search fails
        search_term = k
        org_id = "9606"

        if k == 'kith': 
            search_term = 'TK' 
            org_id = "10298" # HSV-1 Thymidine Kinase
        
        acc, seq = fetch_uniprot_sequence(search_term, organism_id=org_id)
        if acc and seq:
            results.append({"Name": k, "UniProtID": acc, "Sequence": seq})
        
        time.sleep(0.2) # Be nice to the API

    # 3. Write Sequences to CSV
    if not results:
        logger.error("No sequences fetched. Exiting.")
        sys.exit(1)

    df_seq = pd.DataFrame(results)
    df_seq.to_csv(args.output_seq, index=False)
    logger.info(f"Saved {len(df_seq)} sequences to {args.output_seq}")

    if len(df_seq) < 2:
        logger.warning("Not enough sequences to calculate similarity.")
        return

    # 4. Calculate Similarity
    df_sim = calculate_similarity_matrix(results)

    # 5. Write Similarity Matrix
    df_sim.to_csv(args.output_sim, index=False)
    logger.info(f"Saved similarity matrix to {args.output_sim}")

if __name__ == "__main__":
    main()