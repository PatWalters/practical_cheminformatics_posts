# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "datasets==4.3.0",
#     "marimo",
#     "matplotlib==3.10.7",
#     "numpy==2.3.4",
#     "pandas==2.3.3",
#     "rdkit==2025.9.1",
#     "requests==2.32.5",
#     "scikit-learn==1.7.2",
#     "seaborn==0.13.2",
#     "tqdm==4.67.1",
#     "useful-rdkit-utils==0.92",
#     "bblean @ git+https://github.com/mqcomplab/bblean",
# ]
# ///

import marimo

__generated_with = "0.17.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performing Exploratory Data Analysis on the OpenADMET ExpansionRx Blind Challenge Dataset

        When encountering a new dataset, many people quickly jump into building a machine learning model. I prefer to start with exploratory analysis to gain a better understanding of the data. This notebook performs initial exploratory data analysis (EDA) on the OpenADMET ExpansionRx Blind Challenge Dataset. Instead of using Jupyter for this analysis, I'm using **marimo**, a new data science notebook environment that allows you to create interactive data apps with minimal code. I think of **marimo** as a "better Jupyter," because it has several features that make b\
    uilding interactive data apps easier, including built-in support for Altair charts, simple layout options, and interactive widgets. I plan to publish a series of blog posts explaining **marimo's** capabilities and how it can be used for data science and cheminformatics. I'm also work
    ing on a repository titled "Practical Cheminformatics with Marimo," which showcases some ways to use **marimo** for cheminformatics tasks. Please consider this notebook a preview of what's to come. For those interested in learning more about **marimo**, I recommend starting with the
    following resources.

        - Marimo Documentation: https://marimo.readthedocs.io/en/latest/
        - Marimo YouTube Channel: https://www.youtube.com/c/MarimoDataScience

        There are a few aspects of **marimo** that might confuse some long-time Jupyter users (including me).
        1. The output of a code cell in **marimo** appears above the cell in the notebook instead of below it, as in Jupyter.
        2. A **marimo** notebook is reactive, which means that when you change the value of a variable, any code cells that depend on that variable will automatically update. This is different from Jupyter, where you need to manually re-run code cells to see the updated output. This also\
     means that a variable can only be defined once in a **marimo** notebook.
        3. When you run a **marimo** notebook, it first checks to see if you have the necessary libraries installed. If you don't, **marimo** will ask if you'd to install them, and install them for you. This makes it easy to share **marimo** notebooks with others without worrying about d\
    ependencies.
        4. When you open an existing **marimo** notebook, it runs the code in all the cells. As a result, it can take some time to start. Wait for the spinning hourglass in the upper left corner to disappear.

        If you're viewing the static HTML version of this notebook, you're missing out on the interactivity that **marimo** offers. To experience the full features, please consider running this notebook in a **marimo** environment. It's easy to do:
        1. Download this notebook from GitHub. Note that **marimo** notebooks are simply Python files with a .py extension.
        2. Install marimo using the following command:
        ```bash
        pip install marimo
        ```
        3. Use the `marimo` command to run the notebook:
        ```bash
        marimo edit expansion_data_analysis.py --sandbox
        ```
        This command installs all the dependencies and launches the marimo notebook in a sandboxed environment.
        4. Enjoy!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Import the libraries we will need for the analysis.
    This notebook uses [BitBIRCH-Lean](https://www.biorxiv.org/content/10.1101/2025.10.22.684015v1) - a more memory efficient version of the [BitBIRCH](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00030k) clustering method developed by the [Miranda-Quintan research group at the University of Florida](https://quintana.chem.ufl.edu/).
    If you run this notebook with the `--sandbox` flag, the dependencies will be installed for you.

    As a first step, we'll grab a function I wrote to wrap bbllean from GitHub.
    """)
    return


@app.cell
def _():
    try:
        from bblean_cluster import bblean_cluster
    except (ModuleNotFoundError, FileNotFoundError):
        import requests
        cluster_url = "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_posts/refs/heads/main/expansion_data_exploration/bblean_cluster.py"
        response = requests.get(cluster_url)
        response.raise_for_status()
        with open("bblean_cluster.py","w") as ofs:
            ofs.write(response.text)
    return (bblean_cluster,)


@app.cell
def _():
    import pandas as pd                          # dealing with tabular data
    import matplotlib.pyplot as plt              # plotting and arranging plots
    import seaborn as sns                        # plots
    import useful_rdkit_utils as uru             # cheminformatics and data handling utilities
    import numpy as np                           # numerical operations
    from rdkit import Chem                       # cheminformatics
    from sklearn.decomposition import PCA        # principal component analysis
    from itertools import combinations           # generating combinations
    from rdkit import DataStructs                # for molecular similarity
    from tqdm.auto import tqdm                   # progress bars
    import marimo as mo                          # marimo notebook environment
    import altair as alt                         # interactive plotting
    from datasets import load_dataset            # loading datasets from HuggingFace
    return (
        Chem,
        DataStructs,
        PCA,
        alt,
        combinations,
        load_dataset,
        mo,
        np,
        pd,
        plt,
        sns,
        tqdm,
        uru,
    )


@app.cell
def _():
    # set a notebook global to determine whether progress bars are displayed.  I mainly did this to make the HTML version look nicer. 
    hide_progress = True
    return (hide_progress,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Read the Data
    Let's take an initial look at the data. There are a few things to note:
    - Not every compound was tested in every assay. Note that LogD, KSOL, and several other assays have empty values in the first few rows.
    - Marimo automatically creates frequency histograms for numerical columns when displaying dataframes. Unfortunately, the histograms don't appear in the HTML version of the notebook.
    """)
    return


@app.cell
def _(load_dataset):
    train_ds = load_dataset("openadmet/openadmet-expansionrx-challenge-train-data")
    train_df = train_ds["train"].to_pandas()
    train_df
    return (train_df,)


@app.cell
def _(load_dataset):
    test_ds = load_dataset("openadmet/openadmet-expansionrx-challenge-test-data-blinded")
    test_df = test_ds["test"].to_pandas()
    test_df
    return (test_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Adjust Column Names
    I'm a bit picky about how columns are named, so I made couple of quick adjustments to the column names to make them more consistent.
    """)
    return


@app.cell
def _(train_df):
    train_df.rename(columns={"smiles" : "SMILES", "Molecule Name" : "Name"},inplace=True)
    train_df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Log Transform the Assay Values
    Applying a logarithmic transformation to data before building a machine learning model is a common technique used to address issues with data distribution, linearity, and outliers, which can ultimately improve model performance and reliability. Logarithmic transformations can help make skewed data more normally distributed, enhancing the performance of algorithms that rely on this assumption, such as linear regression and k-nearest neighbors. Log transformations can also help linearize relationships between features and the target variable. Many machine learning models assume a linear relationship between input features and the target. If the relationship is non-linear, applying a log transformation can help linearize it, making it easier for models to capture patterns in the data. Outliers can disproportionately influence model training and predictions. Log transformations compress the data scale, reducing the impact of extreme values and making models more robust to outliers. Overall, applying a logarithmic transformation can lead to improved model accuracy, stability, and interpretability by addressing key challenges related to data distribution and relationships. In the cell below, we add 0.001 to values to avoid taking the log of zero.

    Note that LogD is already in log form, so we don't need to transform it.
    """)
    return


@app.cell
def _(np, train_df):
    for col in train_df.columns[3:]:
        train_df[f"Log_{col}"] = np.log10(train_df[col]+0.001)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a list called `assay_cols` containing the names of the log transformed assay columns in `train_df`.
    """)
    return


@app.cell
def _(train_df):
    assay_cols = [x for x in train_df.columns if x.startswith("Log")]
    assay_cols
    return (assay_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Get an Overview of the Data

    As mentioned above, not every compound was tested in every assay.  The following table summarizes the number of records available for each assay.  Note that LogD has the most records, with 5039 out of 5236 compounds having a LogD value. Some other assays have significantly fewer records, with Log_MGMB (log of Mouse Gastrocnemius Muscle Binding) having only 222 records.  Note that, once again, marimo makes it easy to sort the data. You can also use the bottons below the table to generate graphs without writing any code.
    """)
    return


@app.cell
def _(assay_cols, pd, train_df):
    res = []
    train_df_size = len(train_df)
    for icol in train_df[assay_cols].columns:
        num_records = len(train_df.dropna(subset=icol))
        res.append([icol,num_records,num_records/train_df_size])
    pd.DataFrame(res,columns=["Column","# Records","Fraction of Total"]).round(2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5. Check datatypes
    Before performing any sort of anaylsis, it's essential to check that the data is of the correct type.  I can't tell you how many times I've seen numerical data stored as strings, or categorical data stored as numbers.  This can lead to all sorts of problems down the line when trying to analyze the data.  In the cell below, we check the datatypes of the assay columns to ensure they are all numerical.
    """)
    return


@app.cell
def _(assay_cols, train_df):
    train_df[assay_cols].dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6. Examine Data Distributions
    Next, I like to examine the data distributions for the assays I'll be predicting.  Since we've already log transformed the data, it's straightforward to construct histogram of the distributions. When I look at the distribtions, I try get a feel for a few things.
    - Is the data normally distributied or skewed?
    - What is the dynamic range? If it's less than 3 logs, regression may be difficult.
    - Does the range appear realistic, do I see any odd outliers?

    Examining the plots below, we can see that Log_KSOL and Log_Caco-2 Permeability Papp A>B have skewed distributions.  We can also see that several of the assays including LogHLM CLint, Log_Caco-2 Permeability Papp A>B, and the protein and tissue binding assays on the bottom row have a limited dynamic range, which may make regression difficult.
    """)
    return


@app.cell(hide_code=True)
def _(assay_cols, plt, sns, train_df):
    figure, axes = plt.subplots(3,3,figsize=(10,5))
    axes = axes.flatten()
    for i,coli in enumerate(train_df[assay_cols]):
        sns.histplot(train_df[coli],ax=axes[i])
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7. Calculate the Maximum Possible Correlation
    Before building a machine  model, it's important to understand how the relationship between the data distribution and experimental error impacts our ability to build a model.  One means of estimating model  performance on a particular dataset comes from a [2009 paper by Brown, Muchmore and Hajduk](https://www.sciencedirect.com/science/article/abs/pii/S1359644609000403).  In the paper, the authors propose a straightforwad simultation appropch to estimating the maximum possible regression performance with a paraticular dataset.  In this approach, we begin with a dataset, add normally distributed random noise, and calculate the correlation between the original data and the data plus noise.  I wrote more extensive description of the method in a [2019 blog post](https://practicalcheminformatics.blogspot.com/2019/07/how-good-could-should-my-models-be.html).

    The code below uses an implementation in the [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils) library that enables this test with one line of code. Examining the table below, we can see that some of the datasets, with `Max Possible Correlation` near 0.5, may be challenging to model.
    """)
    return


@app.cell
def _(assay_cols, np, pd, tqdm, train_df, uru):
    max_cor_res = []
    for a in tqdm(assay_cols,disable=True):
        vals = np.isfinite(train_df[a])
        max_cor_res.append([a,uru.max_possible_correlation(vals)])
    max_cor_df = pd.DataFrame(max_cor_res,columns=["Assay","Max Possible Correlation"])
    max_cor_df.round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8. Calculate Chemical Fingerprints
    We now add two fingerprint columns to the dataframe.
    - `fp` - a binary fingerprint as a numpy array.  We can use this fingerprint for clustering or machine learning models.
    - `morgan` - an RDKit Morgan fingerprint suitable for molecular similarlity calculations.

    We use the Smi2Fp class from [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils) to simplify fingerprint calculations.
    """)
    return


@app.cell
def _(train_df, uru):
    smi2fp = uru.Smi2Fp()
    train_df['fp'] = train_df.SMILES.apply(smi2fp.get_np)
    train_df['morgan'] = train_df.SMILES.apply(smi2fp.get_fp)
    return (smi2fp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9. Cluster Structures to Better Understand the Chemistry
    Clustering provides an excellent means of grouping similar structures and understanding the composition of, and releationships within, a dataset. While clustering is an informative technique, it can also be slow.  Methods like the Butina algorithm in the RDKit scale as the square of the number of compounds. A recently developed clustering method [BitBIRCH](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00030k), from the [Miranda-Quintana](https://quintana.chem.ufl.edu/) group at the University of Florida has dramatically improved scaling behavior and consequently is much faster. BitBIRCH is able to cluster our training dataset, with more than 5,000 compounds, in a few seconds. The function `bblean_cluster` below is a simple wrapper I wrote around one of the examples in the [bblean GitHub](https://github.com/mqcomplab/bblean)
    """)
    return


@app.cell
def _(bblean_cluster, train_df):
    train_df['cluster'],_ = bblean_cluster(train_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once we've done the clustering, it's helpful to look at the sizes of the clusters.  We can do this using the [value_counts_df](https://useful-rdkit-utils.readthedocs.io/en/latest/pandas.html#useful_rdkit_utils.pandas_utils.value_counts_df) function from [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils).  This function creates a dataframe showing the counts of unique values in a specified column.  Here, we use it to count the number of compounds in each cluster.  We then filter the dataframe to show only clusters with 10 or more compounds.
    """)
    return


@app.cell
def _(train_df, uru):
    cluster_count_df = uru.value_counts_df(train_df,"cluster").query("count >= 10")
    cluster_count_df
    return (cluster_count_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To make it easy to select the clusters with more than 10 members, we'll put the cluster ids, and corresponding cluster sizes, into a dictionary.
    """)
    return


@app.cell
def _(cluster_count_df):
    cluster_count_dict = cluster_count_df.set_index("cluster")['count'].to_dict()
    big_cluster_set = set(cluster_count_df.query("count >= 10").cluster)
    return (cluster_count_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In subsequent cells, we will use Principal Component Analysis (PCA) to reduce the dimensionality of the fingerprint data and visualize the chemical space covered by the dataset. PCA helps us identify patterns and clusters in the data, which can provide insights into the relationships between molecular structures and their associated assay values. In this function, we use a recently published varaiation from the Miranda-Quintana group which normalizes the fingerprints before applying PCA. In their paper, the authors show that this method captures the chemical space more effectively than tSNE or UMAP. The function below takes a list of fingerprints as input and returns a dataframe with the principal components.  If a PCA model is provided, it uses that model to transform the data; otherwise, it fits a new PCA model.
    """)
    return


@app.cell
def _(PCA, np, pd):
    def compute_pca(data, model=None, columns=['PC1', 'PC2']):
        if len(data) < 2:
            return pd.DataFrame()
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        data = data / norms
        if model == None:
            model = PCA(n_components=2)    
            transformed = model.fit_transform(data)
        else:
            transformed = model.transform(data)
        return model, pd.DataFrame(transformed, columns=columns)
    return (compute_pca,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we calcuate the principal components from the `fp` column of the training dataframe.
    """)
    return


@app.cell
def _(PCA, np, train_df):
    X = np.stack(train_df.fp)
    #variances = np.var(np.stack(train_df.fp), axis=0)
    #non_zero_variance_cols = variances > 1e-9
    #X = X[:, non_zero_variance_cols]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    pca_model = PCA(n_components=2)
    X_pca = pca_model.fit_transform(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we create a new dataframe, `big_cluster_df`, containing only the clusters with 10 or more members. This will allow us to focus our analysis on the larger clusters, which are likely to be more informative. We then display the dataframe, excluding the fingerprint columns for clarity.
    """)
    return


@app.cell
def _(cluster_count_dict, compute_pca, np, train_df):
    big_cluster_df = train_df.query("cluster in @big_cluster_set").copy()
    unique_cluster_df = big_cluster_df.drop_duplicates(subset=["cluster"]).copy()
    unique_cluster_df.reset_index(drop=True,inplace=True)
    pca_df = compute_pca(np.stack(unique_cluster_df.fp,dtype=np.int64))[1]
    unique_cluster_df["PC1"] = pca_df.PC1
    unique_cluster_df["PC2"] = pca_df.PC2
    unique_cluster_df["size"] = [cluster_count_dict[x] for x in unique_cluster_df.cluster]
    return big_cluster_df, unique_cluster_df


@app.cell
def _(big_cluster_df):
    cols_to_show = [x for x in big_cluster_df.columns if x not in ['fp','morgan']]
    big_cluster_df[cols_to_show]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we've clustered the data, we can examine the distribution of assay values within each cluster. The boxplot below shows the distribution of values for a selected assay across different clusters. This visualization helps us understand how assay values vary within and between clusters, which can provide insights into the relationships between molecular structures and their associated assay values.  This visualization demonstrates one of the helpful features of marimo - the ability to create interactive widgets with minimal code. In this case, we use a dropdown widget to select the assay to display in the boxplot. When the user selects a different assay, the boxplot automatically updates to show the new data.
    """)
    return


@app.cell
def _(assay_cols, mo):
    boxplot_col = mo.ui.dropdown(options=assay_cols,label="Select assay for boxplot",value=assay_cols[0])
    boxplot_col
    return (boxplot_col,)


@app.cell
def _(big_cluster_df, boxplot_col, sns):
    sns.set(rc={"figure.figsize":(18,5)})
    sns.set_style("whitegrid")
    ax = sns.boxplot(x="cluster",y=boxplot_col.value,data=big_cluster_df)
    # Get all current labels and tick positions
    all_labels = [t.get_text() for t in ax.get_xticklabels()]
    all_ticks = ax.get_xticks()

    # 3. Select every fifth label and corresponding tick position
    # Slicing with [::5] selects every 3rd element starting from the first (index 0)
    fifth_labels = all_labels[::5]
    fifth_ticks = all_ticks[::5]

    # Apply the new ticks and labels to the x-axis
    ax.set_xticks(fifth_ticks)
    ax.set_xticklabels(fifth_ticks)
    return


@app.cell
def _(unique_cluster_df, uru):
    unique_cluster_df['image'] = unique_cluster_df.SMILES.apply(uru.smi_to_base64_image,target="altair")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10. Visualize the Underlying Chemical Space

    Another common practice when performing exploratory data analysis is to create a 2D projection of the chemical space covered by the cluster centers in the dataset. This helps visualize the diversity of the compounds and identify any clusters or patterns that may exist. In this notebook, we use Principal Component Analysis (PCA) to reduce the dimensionality of the fingerprint data and create a 2D projection. The scatter plot below shows the PCA projection of the compounds in the dataset, with each point representing a compound. The axes represent the first two principal components, which capture the most variance in the data. By examining this plot, we can gain insights into the chemical diversity of the dataset and identify any potential clusters or outliers that may warrant further investigation.  The code below presents an interactive view of the chemical space. Holding the mouse over a point shows the structure of a representative molecule from a cluster. By selecting one or more clusters on the left, the structures in those clusters are displayed on the right.
    """)
    return


@app.cell
def _(alt, mo, unique_cluster_df):
    chart = mo.ui.altair_chart(alt.Chart(unique_cluster_df[["PC1","PC2","image"]]).mark_circle(size=60).encode(
        x = alt.X('PC1',title='PC1',
            axis=alt.Axis(
                titleFontSize=16,  # Font size for the x-axis title
                labelFontSize=12   # Font size for the x-axis tick labels (numbers/text)
            )),
        y = alt.Y('PC2',title='PC2',
            axis=alt.Axis(
                titleFontSize=16,  # Font size for the x-axis title
                labelFontSize=12   # Font size for the x-axis tick labels (numbers/text)
            )),
        tooltip = alt.Tooltip(["image"])
        ))
    return (chart,)


@app.cell
def _(Chem, chart, train_df, unique_cluster_df):
    idx = chart.value.index
    current_clusters = unique_cluster_df.iloc[idx].cluster
    if len(current_clusters) > 0:
        selected_df = train_df.query("cluster in @current_clusters").head(10)
        cluster_mol_list = [Chem.MolFromSmiles(x) for x in selected_df.SMILES]
        legends = [f"{name} | {cluster}" for name,cluster in zip(selected_df.Name,selected_df.cluster)]
        mol_img = Chem.Draw.MolsToGridImage(cluster_mol_list,molsPerRow=5,subImgSize=(200,200),legends=legends)
    else:
        mol_img = "No cluster selected"
    return (mol_img,)


@app.cell
def _(chart, mo, mol_img):
    mo.hstack([chart,mol_img],widths='equal')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 11. Look for Activity Cliffs
    When exploring a chemical dataset, it's always important to look for activity cliffs. These are cases where a small change in chemical structure brings about a large change in a property or biological activity.  These activity cliffs can help us to identify structural features that are key activity drivers. Activity cliffs can also be useful in identifying assay errors or artifacts.  In the cells below we'll use the Structure Activity Landscape Index (SALI) to identify activity cliffs.  The SALI value is simply the difference in activity between two compounds divided by 1.0 minus their Tanimoto similarity. Note that we add 0.001 to the similarity difference to avoid divide by zero errors.
    """)
    return


@app.cell
def _(DataStructs, combinations, hide_progress, np, pd, tqdm, train_df):
    def calc_sali(df,assay):
        res_list = []
        for k,v in tqdm(train_df.groupby("cluster"),desc=assay,disable=hide_progress):
            v = v.dropna(subset=assay).copy()
            v.sort_values(assay,inplace=True)
            for (_,row_1),(_,row_2) in combinations(v.iterrows(),2):
                delta = row_1[assay] - row_2[assay]
                sim = DataStructs.TanimotoSimilarity(row_1.morgan,row_2.morgan)
                sali = np.abs(delta)/(1-sim+0.001)
                res_list.append([row_1.SMILES,row_1.Name,row_1[assay],row_2.SMILES,row_2.Name,row_2[assay],sim,sali])
        sali_df = pd.DataFrame(res_list,columns=["SMILES_1","Name_1","Val_1","SMILES_2","Name_2","Val_2","Tanimoto","SALI"])
        sali_df.sort_values("SALI",ascending=False,inplace=True)
        return sali_df
    return (calc_sali,)


@app.cell
def _(assay_cols, mo):
    menu = mo.ui.dropdown(options = assay_cols, label="Select an assay", value=assay_cols[0])
    menu
    return (menu,)


@app.cell
def _(calc_sali, menu, train_df):
    sali_df = calc_sali(train_df,menu.value)
    return (sali_df,)


@app.cell
def _(Chem, np, sali_df):
    num_to_show = 12
    mol_list = [Chem.MolFromSmiles(x) for x in np.array(list(zip(sali_df.SMILES_1,sali_df.SMILES_2))).flatten()[:num_to_show]]
    label_list = [f"{x:.1f}" for x in np.array(list(zip(sali_df.Val_1,sali_df.Val_2))).flatten()]
    return label_list, mol_list


@app.cell
def _(Chem, label_list, mol_list, uru):
    uru.rd_make_structures_pretty()
    cliff_img = Chem.Draw.MolsToGridImage(mol_list,molsPerRow=4,subImgSize=(300,300),legends=label_list)
    return (cliff_img,)


@app.cell
def _(cliff_img, menu, mo):
    header_md = mo.md(f"### **Activity Cliffs for {menu.value}**")
    mo.vstack([header_md,cliff_img])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The visualization above provides a good way of looking at specific activity cliffs, but it's also helpful to look at a more complete view of the activity landscape.  One way to do this is to calculate the overall fraction of compounds pairs containing activity ciffs. In the example below, we will consider any pair of compounds with Tanimoto similarity > 0.5 and a difference in activity > 1 log to be an activity cliff.  To create the plot below, we begin by calculating the SALI for all pairs for each assay. We can the use the subsequent dataframes to create the plot below.
    """)
    return


@app.cell
def _(assay_cols, calc_sali, train_df):
    df_list = []
    for assay_col in assay_cols:
        tmp_df = calc_sali(train_df,assay_col)
        tmp_df['assay'] = assay_col
        df_list.append(tmp_df)
    return (df_list,)


@app.cell
def _(assay_cols, df_list, np):
    assay_res = []
    for name, df in zip(assay_cols,df_list):
        num_pairs = len(df.query("Tanimoto >= 0.5"))
        df['abs_delta'] = np.abs(df.Val_1 - df.Val_2)
        num_cliffs = len(df.query("Tanimoto >= 0.5 and abs_delta >= 1"))
        assay_res.append([name,num_cliffs/num_pairs])
    return (assay_res,)


@app.cell
def _(assay_res, pd, sns):
    res_df = pd.DataFrame(assay_res,columns=["Assay","Cliffs"])
    sns.barplot(y="Assay",x="Cliffs",data=res_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 12. Compare the Training and Test Sets
    Before emabarking on any modeling expercise, it's a good idea to calculate the similarity of the molecules in the training set to those in the test set.  I wrote a function `compare_datasets` in [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils) to make this easy.
    """)
    return


@app.cell
def _(assay_cols, hide_progress, pd, smi2fp, test_df, tqdm, train_df, uru):
    # calculate morgan fingerprints for the test set
    test_df['morgan'] = test_df.SMILES.apply(smi2fp.get_fp)
    sim_df_list = []
    for acol in tqdm(assay_cols,disable=hide_progress):
        # drop training set rows without data
        tmp_sim_df = train_df.dropna(subset=acol)
        max_sim = uru.compare_datasets(tmp_sim_df.morgan, test_df.morgan)
        sim_df = pd.DataFrame({"assay" : acol, "sim" : max_sim })
        sim_df_list.append(sim_df)
    return (sim_df_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can plot the train/test similarity as boxplots.  I  find it easier when I do the box plots horizontally. From the boxplots we can see that the median train/test similarity is around 0.5 for most of the assays. It's lower for Log_MGMB where we have fewer datapoints.
    """)
    return


@app.cell
def _(pd, sim_df_list, sns):
    sim_ax = sns.boxplot(y="assay",x="sim",data=pd.concat(sim_df_list),color="lightblue")
    sim_ax.set_xlabel("Tanimoto Similarity of Test Set to Training Set")
    sim_ax.set_ylabel("Assay")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conclusion
    I hope this notebook has shown you a few tricks for EDA and has demonstrated some of the cool things you can do with **marimo**.  A lot of the code here will also work in Jupyter, but you won't get the nifty interactive views.  As I mentioned at the begining, I have some additional **marimo** notebooks for Cheminformatics in the works.  Those should be out in a few weeks.

    ### Acknowledgements

    I wouldn't have tried **marimo** if it weren't for blogs by [Eric Ma](https://ericmjl.github.io/blog/) and [Srijit Seal](https://srijitseal.com).  Those guys are a constant source of inspiration.  Thanks to Ramon Miranda-Quintana, Ignacio Pickering, and Kenneth Lopez Perez for their help with BitBIRCH and BBLean. Speical thanks to the **marimo** team.  They've created an amazing tool, their support is fantastic, and [Vincent's videos](https://www.youtube.com/@marimo-team) are the best!
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
