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
