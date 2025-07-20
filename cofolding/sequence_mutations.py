import pandas as pd
import requests

def get_smiles_from_pdb_cid(pdb_cid: str) -> str | None:
    """
    Fetches the Canonical SMILES string for a given PDB Chemical Component ID (CID)
    using the RCSB PDB Data API.

    PDB CIDs are 3-letter codes (e.g., "ATP", "HEM") that identify small molecules
    (ligands, ions, etc.) within PDB structures.

    Args:
        pdb_cid: The 3-letter PDB Chemical Component ID (e.g., "ATP", "HEM").

    Returns:
        The Canonical SMILES string if found, otherwise None.
    """
    # RCSB PDB Data API endpoint for chemical components
    # We are querying the 'chemcomp' (chemical component) endpoint with the PDB CID.
    # The API returns a JSON object containing various details, including chemical descriptors.
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{pdb_cid.upper()}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Navigate the JSON structure to extract the Canonical SMILES.
        # The SMILES are typically found under 'pdbx_chem_comp_descriptor'
        # with a 'type' of 'SMILES' and a 'program' like 'OpenEye OEToolkits'
        # or 'CACTVS' for canonical SMILES.
        if "pdbx_chem_comp_descriptor" in data:
            for descriptor in data["pdbx_chem_comp_descriptor"]:
                # Look for the Canonical SMILES descriptor.
                # RCSB PDB provides multiple SMILES types; Canonical SMILES is usually preferred.
                # We'll prioritize OpenEye OEToolkits or CACTVS for canonical.
                if descriptor.get("type") == "SMILES" and descriptor.get("program") in ["OpenEye OEToolkits", "CACTVS"]:
                    # Check if it's a canonical SMILES specifically
                    if "Canonical" in descriptor.get("descriptor", ""): # Check for "Canonical" in the descriptor string itself
                         return descriptor.get("descriptor")
                    elif "CanonicalSMILES" in descriptor.get("descriptor", ""): # Some might be directly named CanonicalSMILES
                        return descriptor.get("descriptor")
                    # Fallback to any SMILES if canonical isn't explicitly tagged in the descriptor string
                    # This might not be strictly canonical but provides a SMILES.
                    elif "SMILES" in descriptor.get("type"):
                        return descriptor.get("descriptor")
        return None

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            print(f"PDB CID '{pdb_cid}' not found in RCSB PDB Chemical Component Dictionary.")
        else:
            print(f"HTTP error occurred: {http_err} - Could not retrieve SMILES for PDB CID '{pdb_cid}'")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err} - Check your internet connection.")
        return None
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err} - Request timed out.")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def convert_amino_acid_abbr(three_letter_abbr: str) -> str:
    """
    Converts a three-letter amino acid abbreviation to its one-character equivalent.

    Args:
        three_letter_abbr: A string representing the three-letter amino acid abbreviation
                           (e.g., "Ala", "Gly", "Lys"). Case-insensitive.

    Returns:
        A string representing the one-character amino acid equivalent.
        Returns "Unknown" if the abbreviation is not found in the mapping.

    Examples:
        >>> convert_amino_acid_abbr("Ala")
        'A'
        >>> convert_amino_acid_abbr("gly")
        'G'
        >>> convert_amino_acid_abbr("Lys")
        'K'
        >>> convert_amino_acid_abbr("XYZ")
        'Unknown'
    """
    # Create a dictionary mapping three-letter abbreviations to one-character codes
    # All keys are stored in uppercase for case-insensitive lookup
    amino_acid_map = {
        "ALA": "A",  # Alanine
        "ARG": "R",  # Arginine
        "ASN": "N",  # Asparagine
        "ASP": "D",  # Aspartic Acid
        "CYS": "C",  # Cysteine
        "GLN": "Q",  # Glutamine
        "GLU": "E",  # Glutamic Acid
        "GLY": "G",  # Glycine
        "HIS": "H",  # Histidine
        "ILE": "I",  # Isoleucine
        "LEU": "L",  # Leucine
        "LYS": "K",  # Lysine
        "MET": "M",  # Methionine
        "PHE": "F",  # Phenylalanine
        "PRO": "P",  # Proline
        "SER": "S",  # Serine
        "THR": "T",  # Threonine
        "TRP": "W",  # Tryptophan
        "TYR": "Y",  # Tyrosine
        "VAL": "V",  # Valine
        # Common ambiguous or special codes
        "ASX": "B",  # Asparagine or Aspartic Acid
        "GLX": "Z",  # Glutamine or Glutamic Acid
        "XAA": "X",  # Any amino acid
        "SEC": "U",  # Selenocysteine
        "PYL": "O",  # Pyrrolysine
    }

    # Convert the input abbreviation to uppercase to ensure case-insensitivity
    upper_abbr = three_letter_abbr.upper()

    # Look up the one-character code in the map, return "Unknown" if not found
    return amino_acid_map.get(upper_abbr, "Unknown")

def mutate_sequence(atm_grp, res_num_list, new_res=None):
    miyata_dict = {'Y': 'G',
                   'W': 'G',
                   'K': 'G',
                   'R': 'G',
                   'V': 'D',
                   'L': 'D',
                   'I': 'D',
                   'M': 'D',
                   'F': 'D',
                   'A': 'W',
                   'G': 'W',
                   'P': 'W',
                   'H': 'W',
                   'D': 'W',
                   'E': 'W',
                   'C': 'W',
                   'N': 'W',
                   'Q': 'W',
                   'T': 'W',
                   'S': 'W'}
    prot = atm_grp.select("protein")
    df = pd.DataFrame({"resnum" : prot.getResnums(),
                       "resname" : prot.getResnames()})
    df.drop_duplicates(subset="resnum",inplace=True)
    df.set_index(df.resnum, inplace=True)
    df['single'] = df.resname.apply(convert_amino_acid_abbr)
    if new_res is not None:
        for r in res_num_list:
            current_res = df.loc[r,'single']
            if new_res == 'miyata':
                replacement = miyata_dict[current_res]
            else:
                replacement = new_res
            df.loc[r,'single'] = replacement
    return "".join(df.query("single != 'Unknown'").drop_duplicates(subset="resnum").single.values)
