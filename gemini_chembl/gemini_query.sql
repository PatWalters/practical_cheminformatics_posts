.headers on
.mode csv
WITH RECURSIVE KinaseHierarchy(protein_class_id) AS (
    SELECT protein_class_id
    FROM protein_classification
    WHERE pref_name = 'Kinase'
    UNION ALL
    SELECT pc.protein_class_id
    FROM protein_classification pc
    JOIN KinaseHierarchy kh ON pc.parent_id = kh.protein_class_id
)
SELECT DISTINCT
    cs.canonical_smiles,
    act.molregno,
    td.chembl_id AS target_chembl_id,
    act.activity_id,
    td.pref_name AS target_name,
    d.year,
    d.doi AS article_doi,
    act.standard_value AS IC50,
    act.standard_units AS units
FROM activities act
JOIN assays ass ON act.assay_id = ass.assay_id
JOIN docs d ON ass.doc_id = d.doc_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN compound_structures cs ON act.molregno = cs.molregno
JOIN target_components tc ON td.tid = tc.tid
JOIN component_class cc ON tc.component_id = cc.component_id
JOIN KinaseHierarchy kh ON cc.protein_class_id = kh.protein_class_id
WHERE
    d.year > 2022
    AND td.target_type = 'SINGLE PROTEIN'
    AND td.organism = 'Homo sapiens'
    AND act.standard_type = 'IC50';
