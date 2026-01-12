import sqlite3
import csv

# Connect to database
conn = sqlite3.connect('chembl_36/chembl_36_sqlite/chembl_36.db')
cursor = conn.cursor()

# Execute query
query = """
SELECT DISTINCT
    cs.canonical_smiles AS smiles,
    md.chembl_id AS chembl_id,
    td.pref_name AS target_name,
    d.year AS publication_year,
    d.doi AS article_doi,
    a.standard_value AS ic50_value,
    a.standard_units AS ic50_units
FROM activities a
    INNER JOIN assays ass ON a.assay_id = ass.assay_id
    INNER JOIN target_dictionary td ON ass.tid = td.tid
    INNER JOIN target_components tc ON td.tid = tc.tid
    INNER JOIN component_class cc ON tc.component_id = cc.component_id
    INNER JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
    INNER JOIN molecule_dictionary md ON a.molregno = md.molregno
    INNER JOIN compound_structures cs ON md.molregno = cs.molregno
    LEFT JOIN docs d ON a.doc_id = d.doc_id
WHERE 
    a.standard_type = 'IC50'
    AND d.year > 2022
    AND pc.protein_class_desc LIKE '%kinase%'
    AND a.standard_value IS NOT NULL
ORDER BY d.year DESC, md.chembl_id;
"""

cursor.execute(query)

# Write to CSV with proper quoting
with open('kinase_inhibitors_after_2022.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    
    # Write header
    writer.writerow(['smiles', 'chembl_id', 'target_name', 'publication_year', 'article_doi', 'ic50_value', 'ic50_units'])
    
    # Write data
    for row in cursor:
        writer.writerow(row)

conn.close()
print(f"Data extracted successfully!")
