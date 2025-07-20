from io import StringIO
import pandas as pd
import numpy as np

def build_miyata_dict():
    miyata_buff = """WT MUT DIST
    TYR GLY 4.08
    TRP GLY 5.13
    LYS GLY 3.54
    ARG GLY 3.58
    VAL ASP 3.40
    LEU ASP 4.10
    ILE ASP 3.98
    MET ASP 3.69
    PHE ASP 4.27
    ALA TRP 4.23
    GLY TRP 5.13
    PRO TRP 4.17
    HIS TRP 3.16
    ASP TRP 4.88
    GLU TRP 4.08
    CYS TRP 3.34
    ASN TRP 4.39
    GLN TRP 3.42
    THR TRP 3.50
    SER TRP 4.38"""

    ios = StringIO(miyata_buff)
    aa_df = pd.read_csv(ios,sep=" ")
    miyata_dict = {(a,b) for a,b in aa_df[["WT","MUT"]].values}
    return miyata_dict
