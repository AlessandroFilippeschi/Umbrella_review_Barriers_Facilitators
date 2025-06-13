#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
st_time = time.time()
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import umap

import tensorflow as tf
print(tf.__version__)

version_fn = getattr(tf.keras, "__version__", None)

if version_fn is not None:
    print("Keras version:", version_fn)
else:
    print("L'attributo '__version__' non esiste in tf.keras")

import tensorflow_hub as hub

# Use spaCy for lemmatization
import spacy
nlp = spacy.load("en_core_web_sm")

# Use HDBSCAN for clustering
import hdbscan

# Use RapidFuzz for fuzzy matching
from rapidfuzz import fuzz

from vocabulary import *

########################################################################
## 2. Text Cleaning & Preprocessing with Fuzzy Standardization
########################################################################

def fuzzy_standardize(text, term_dict, threshold=70):
    """
    Applies fuzzy matching on token n-grams to catch near-matches
    for phrases in term_dict. If an n-gram scores above the threshold
    (using RapidFuzz's ratio), it is replaced with the standardized key.
    """
    tokens = text.split()
    new_tokens = []
    i = 0
    # Process tokens sequentially
    while i < len(tokens):
        replaced = False
        # For each standard term, try each phrase
        for std_term, phrases in term_dict.items():
            for phrase in phrases:
                phrase_words = phrase.split()
                n = len(phrase_words)
                if i + n <= len(tokens):
                    candidate = " ".join(tokens[i:i+n])
                    score = fuzz.ratio(candidate, phrase)
                    if score >= threshold:
                        new_tokens.append(std_term)
                        i += n
                        replaced = True
                        break
            if replaced:
                break
        if not replaced:
            new_tokens.append(tokens[i])
            i += 1
    return " ".join(new_tokens)

def clean_text(text):
    """
    Cleans text by:
      1. Lower-casing
      2. Removing parentheticals and bullet-like prefixes
      3. Standardizing domain-specific synonyms using regex and fuzzy matching
      4. Removing extra punctuation
      5. Lemmatizing using spaCy
      6. Removing stopwords (both default and extended)
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove bullet prefixes or leading numbers
    text = re.sub(r'^\s*\d+[\).-]?\s*', '', text)
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'e\.g\.|i\.e\.', '', text)
    # Exact standardization using regex
    for std_term, phrases in TERM_STANDARDIZATION.items():
        pattern = r'\b(?:' + '|'.join(map(re.escape, phrases)) + r')\b'
        text = re.sub(pattern, std_term, text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Fuzzy matching to catch near-misses
    text = fuzzy_standardize(text, TERM_STANDARDIZATION, threshold=70)
    # Lemmatize and remove stopwords
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        lemma = token.lemma_
        if lemma == "-PRON-":
            lemma = token.text
        if token.is_stop or lemma in EXTENDED_STOPWORDS:
            continue
        tokens.append(lemma)
    return " ".join(tokens)

########################################################################
## 3. Universal Sentence Encoder, Clustering & Post-Processing
########################################################################

def load_use_model():
    print("Loading Universal Sentence Encoder model...")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return model

def embed_texts_use(text_list, use_model):
    embeddings = use_model(text_list)
    embeddings_np = embeddings.numpy()
    embeddings_normalized = normalize(embeddings_np)
    return embeddings_normalized

def assign_cluster_names_tfidf(df, cluster_col="Cluster", cleaned_text_col="cleaned_text", top_n=3):
    """
    Uses TF‑IDF to extract representative terms from each cluster.
    The noise cluster (-1) is labeled as "Noise".
    """
    cluster_names = {}
    unique_clusters = sorted(df[cluster_col].unique())
    for clus_id in unique_clusters:
        if clus_id == -1:
            cluster_names[clus_id] = "Noise"
            continue
        texts = df[df[cluster_col] == clus_id][cleaned_text_col].tolist()
        if not texts:
            cluster_names[clus_id] = f"Topic{clus_id}: misc"
            continue
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            cluster_names[clus_id] = f"Topic{clus_id}: misc"
            continue
        tfidf_means = tfidf_matrix.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        if len(tfidf_means) == 0:
            cluster_names[clus_id] = f"Topic{clus_id}: misc"
            continue
        top_indices = tfidf_means.argsort()[-top_n:][::-1]
        top_terms = [terms[i] for i in top_indices]
        cluster_names[clus_id] = f"Topic{clus_id}: " + " / ".join(top_terms)
    df["ClusterName"] = df["Cluster"].map(cluster_names)
    return df

def merge_similar_clusters(df, embeddings, similarity_threshold=0.9):
    """
    Post-processes clustering results by computing centroids for each non-noise cluster
    and merging clusters whose centroids have cosine similarity above the threshold.
    Uses a union-find method to merge clusters transitively.
    """
    unique_clusters = sorted([c for c in df["Cluster"].unique() if c != -1])
    if not unique_clusters:
        return df  # Nothing to merge if only noise exists
    centroids = {}
    for clus in unique_clusters:
        indices = df.index[df["Cluster"] == clus].tolist()
        centroids[clus] = np.mean(embeddings[indices], axis=0)
    merge_pairs = []
    #print("In merge Similarity threshold is", similarity_threshold)
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            c1 = unique_clusters[i]
            c2 = unique_clusters[j]
            sim = cosine_similarity(centroids[c1].reshape(1, -1), centroids[c2].reshape(1, -1))[0, 0]
            if sim >= similarity_threshold:
                merge_pairs.append((c1, c2))
    # Union-Find structure for merging clusters
    parent = {cid: cid for cid in unique_clusters}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX
    for (c1, c2) in merge_pairs:
        union(c1, c2)
    new_labels = {cid: find(cid) for cid in unique_clusters}
    # Update the df cluster labels; noise (-1) remains unchanged
    df["Cluster"] = df["Cluster"].apply(lambda x: new_labels[x] if x in new_labels else x)
    return df

def cluster_texts_hdbscan(df, text_col, use_model, min_cluster_size=5, min_samples=None, sim_thr=0.9):
    # Reset index to ensure contiguous indexing for proper alignment with embeddings
    df = df.reset_index(drop=True)
    df["cleaned_text"] = df[text_col].apply(clean_text)
    texts = df["cleaned_text"].tolist()
    embeddings = embed_texts_use(texts, use_model)
    print(embeddings.shape)
    n_samples, n_features = embeddings.shape
    n_components = min(n_features, n_samples, 50)  # safeguard for PCA

    UMAP_flag = False
    if UMAP_flag:
        # use UMAP instead of PCA
        UMAP_fit = umap.UMAP(
            n_neighbors=90,
            min_dist=0.55,
            n_components=n_components,
            metric="euclidean"
        )
        embeddings_red = UMAP_fit.fit_transform(embeddings)
    else:
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_red = pca.fit_transform(embeddings)
    print(embeddings_red.shape)
    #print("Min cluster size is: ", min_cluster_size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean')
    labels = clusterer.fit_predict(embeddings_red)
    df["Cluster"] = labels
    # Post-process: merge similar clusters (noise remains as -1)
    df = merge_similar_clusters(df, embeddings_red, similarity_threshold=sim_thr)
    df = assign_cluster_names_tfidf(df, cluster_col="Cluster", cleaned_text_col="cleaned_text")
    return df, clusterer, embeddings_red


def calculate_intra_cluster_similarity(embeddings, labels):
    results = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_emb = embeddings[cluster_indices]
        if len(cluster_emb) > 1:
            sim_matrix = cosine_similarity(cluster_emb)
            np.fill_diagonal(sim_matrix, 0)
            n = len(cluster_emb)
            sum_upper = np.triu(sim_matrix, 1).sum()
            denom = (n * (n - 1)) / 2
            avg_sim = sum_upper / denom if denom > 0 else 0
        else:
            avg_sim = 1.0
        results[label] = avg_sim
    return results

def print_cluster_summary(df, embeddings_pca):
    labels = df["Cluster"].values
    intra_sim = calculate_intra_cluster_similarity(embeddings_pca, labels)
    cluster_info = []
    unique_labels = sorted(df["Cluster"].unique())
    for label in unique_labels:
        count = (labels == label).sum()
        cluster_name = df[df["Cluster"] == label]["ClusterName"].iloc[0]
        sim = intra_sim[label] if label in intra_sim else None
        cluster_info.append({
            "Cluster": label,
            "ClusterName": cluster_name,
            "Count": count,
            "IntraClusterSimilarity": sim
        })
    summary_df = pd.DataFrame(cluster_info)
    return summary_df

def recluster_noise(df, text_col, use_model, min_cluster_size, sim_thr, min_samples=None):
    """
    Extracts noise points (Cluster == -1) and re-clusters them using HDBSCAN.
    """
    noise_df = df[df["Cluster"] == -1].copy()
    if noise_df.empty:
        return None, None, None
    print("Re-clustering noise points...")
    new_df, new_clusterer, new_embeddings = cluster_texts_hdbscan(noise_df, text_col, use_model,
                                                                  min_cluster_size=min_cluster_size,
                                                                  min_samples=min_samples,
                                                                  sim_thr=sim_thr)
    return new_df, new_clusterer, new_embeddings


########################################################################
## 5. Reading & Splitting Excel (Including Income_level)
########################################################################

def read_and_split_excel(excel_file_path: str):
    df = pd.read_excel(excel_file_path, engine="openpyxl")
    required_cols = ["Author", "Year", "Barriers", "Facilitators", "Income_level"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in Excel file.")
    barrier_rows = []
    for idx, row in df.iterrows():
        author = row["Author"]
        year = row["Year"]
        income = row["Income_level"]
        barriers = row["Barriers"]
        if pd.isna(barriers) or str(barriers).strip().upper() == "N/A":
            continue
        for b in str(barriers).split("\n"):
            b = b.strip()
            if b:
                barrier_rows.append((author, year, income, b))
    barriers_df = pd.DataFrame(barrier_rows, columns=["Author", "Year", "Income_level", "Barrier"])

    facilitator_rows = []
    for idx, row in df.iterrows():
        author = row["Author"]
        year = row["Year"]
        income = row["Income_level"]
        facilitators = row["Facilitators"]
        if pd.isna(facilitators) or str(facilitators).strip().upper() == "N/A":
            continue
        for f in str(facilitators).split("\n"):
            f = f.strip()
            if f:
                facilitator_rows.append((author, year, income, f))
    facilitators_df = pd.DataFrame(facilitator_rows, columns=["Author", "Year", "Income_level", "Facilitator"])
    return barriers_df, facilitators_df

########################################################################
## 6. Compute similarities
########################################################################

def opt_compute_similarities(sim_threshold, df, txt, min_clu_sz): 
    print("In Cluster Similarity threshold is", sim_threshold)
    print("Minimum Cluster Size is", min_clu_sz)
    # --- Overall Clustering ---
    df, clusterer, pca = cluster_texts_hdbscan(
        df,
        text_col=txt,
        use_model=use_model,
        min_cluster_size=int(min_clu_sz),
        sim_thr = sim_threshold
    )
    summary = print_cluster_summary(df, pca)
    #print(summary)
    sel_summary = summary.loc[summary['Cluster']>-1]
    sel_summary = sel_summary.loc[sel_summary['IntraClusterSimilarity']>0.05]

    return -sel_summary["Count"].sum()
        

def con_compute_similarities(sim_threshold, df, txt, min_clu_sz, qt, thr_ics): 

    # --- Overall Clustering ---
    df, clusterer, pca = cluster_texts_hdbscan(
        df,
        text_col=txt,
        use_model=use_model,
        min_cluster_size=int(min_clu_sz),
        sim_thr = sim_threshold
    )
    summary = print_cluster_summary(df, pca)
    sel_summary = summary.loc[summary.IntraClusterSimilarity> summary["IntraClusterSimilarity"].min()]
    return sel_summary["IntraClusterSimilarity"].quantile(qt) - thr_ics


def get_computed_similarities(sim_threshold, df, txt, min_clu_sz): 
    # --- Overall Clustering ---
    df, clusterer, pca = cluster_texts_hdbscan(
        df,
        text_col=txt,
        use_model=use_model,
        min_cluster_size=int(min_clu_sz),
        sim_thr = sim_threshold
    )
    summary = print_cluster_summary(df, pca)
    print('\n Summary')
    print(summary)
    return df, summary   


#############################################################################
# OPTIMIZATION
#############################################################################

#if __name__ == "__main__":
excel_file = "C:\Health_loc\\UmbrellaReview\data.xlsx"
out_path = "C:\\Health_loc\\UmbrellaReview\\umbrella_env\\output_files"
use_model = load_use_model()
write_file = False
barriers_df, facilitators_df = read_and_split_excel(excel_file)
#third input either "Barrier" or "Facilitator"
print("\n Time to load models =",  time.time()-st_time)


# initial guess of similarity threshold and of minimum cluster size
x0 = np.array([0.75, int(8)],dtype=object)
bounds = ((0.30,0.99),(int(4),int(18)))
#threshold for intracluster similarity acceptability and quartile
ICS_thr = 0.50
qt=0.5
# Flag for reclustering noise
recluster = False

""" 
opt_out_df, opt_out_sum = get_computed_similarities(0.8,facilitators_df,"Facilitator",4,False)
print('\n\nSummary rc False')
print(opt_out_sum)
noise_count = opt_out_sum['Count'][opt_out_sum['Cluster']==-1].values[0]
print('\nNoise count RC False', noise_count)
opt_out_df, opt_out_sum = get_computed_similarities(0.8,facilitators_df,"Facilitator",4,True)
print('\n\nSummary rc False')
print(opt_out_sum)
noise_count = opt_out_sum['Count'][opt_out_sum['Cluster']==-1].values[0]
print('\nNoise count RC True', noise_count)
val = con_compute_similarities(0.8,facilitators_df,"Facilitator",4,0.5, 0.5, False)
print('\nVal RC false =',val)
val = con_compute_similarities(0.8,facilitators_df,"Facilitator",4,0.5, 0.5, True)
print('\nVal RC true =',val)
"""

res = pd.DataFrame(columns=["Type","Income","Count","Noise","IC_Similarity","Recluster","Quartile","Success","Best_min_size","Best_threshold","Best_min_size_inc","Best_threshold_inc"])
opt_thr = 0.75
opt_mcs = 4
opt_thr_rec = 0.75
opt_mcs_rec = 4

 # Optimization for Facilitators

datasets = ["Facilitator", "Barrier"]
recs = [False, True]

for ds in datasets:

    if ds == "Facilitator":
        dataframe_all = facilitators_df
    else:
        dataframe_all = barriers_df

    income_levels = sorted(dataframe_all["Income_level"].unique())
    income_levels.insert(0,'All')
    print(income_levels)

    for level in income_levels:
        print('\nThis level is: ', level)
        
        if level != 'All':
            print(f"\n=== Clustering ds for Income Level: {level} ===")
            dataframe = dataframe_all[dataframe_all["Income_level"] == level].copy()
            if dataframe.empty:
                continue
        else:
            dataframe = dataframe_all

        for recluster in recs:

            if not recluster:

                #Find best parameters for scanning and merging clusters
                opt_fun = lambda x: opt_compute_similarities(x[0],dataframe, ds, x[1])
                con_fun = lambda x: con_compute_similarities(x[0],dataframe, ds, x[1], qt, ICS_thr)

                con = {'type': 'ineq', 'fun': con_fun}
                solution = minimize(opt_fun,x0,method='SLSQP',bounds=bounds,constraints=con, options={'eps':np.array([0.05, 1])})
                opt_thr = solution.x[0]
                opt_mcs = solution.x[1]
                print("\n Solution = ", solution)
                
                #Get optmal df and summary
                opt_out_df, opt_out_sum = get_computed_similarities(opt_thr,dataframe,ds,opt_mcs)
                
                # Write results
                opt_out_sum.to_csv(os.path.join(out_path,f"{ds}s_recluster_{recluster}_Income_{level}.csv"), index=False)
                opt_out_df[["Author", "Year", "Income_level", ds , "Cluster", "ClusterName"]].to_csv(os.path.join(out_path,f"{ds}s_output_recluster_{recluster}_Income_{level}.csv"), index=False)
                all_count = opt_out_sum['Count'][opt_out_sum['Cluster']>-1].sum()
                noise_count = opt_out_sum['Count'][opt_out_sum['Cluster']==-1].values[0]
                ICS_avg = opt_out_sum['IntraClusterSimilarity'][opt_out_sum['Cluster']>-1].mean()
                newrow = {"Type":ds,"Income":level,"Count":-solution.fun,"Noise":noise_count,"IC_Similarity":ICS_avg,"Recluster":recluster,"Quartile":qt,"Success":solution.success,"Best_min_size":int(opt_mcs),"Best_threshold":opt_thr,"Best_min_size_inc":np.nan,"Best_threshold_inc":np.nan}
                res.loc[len(res)] = newrow
                print("\n Total time =",  time.time()-st_time)
        
            else: #Recluster is True
                
                if (opt_out_df["Cluster"] == -1).any():
                    noise_df = opt_out_df[opt_out_df["Cluster"] == -1].copy()
                    #print(noise_df)
                if noise_df is not None:
                    opt_fun = lambda x: opt_compute_similarities(x[0], noise_df, ds, x[1])
                    con_fun = lambda x: con_compute_similarities(x[0], noise_df, ds, x[1], qt, ICS_thr)
                    solution_rec = minimize(opt_fun,x0,method='SLSQP',bounds=bounds,constraints=con, options={'eps':np.array([0.05, 1])})
                    opt_thr_rec = solution_rec.x[0]
                    opt_mcs_rec = solution_rec.x[1]
                    noise_opt_df, noise_opt_clusterer, noise_opt_pca = cluster_texts_hdbscan(noise_df, text_col=ds, use_model=use_model, min_cluster_size=int(opt_mcs_rec), sim_thr = opt_thr_rec)
                    noise_summary = print_cluster_summary(noise_opt_df, noise_opt_pca)
                    print('\n Noise Summary')
                    print(noise_summary)
                    opt_out_df_nn = opt_out_df.loc[opt_out_df['Cluster']>-1]
                    opt_out_sum_nn = opt_out_sum.loc[opt_out_sum['Cluster']>-1]
                    ntopics = noise_summary.shape
                    opt_out_sum_nn['Cluster'] = opt_out_sum['Cluster'] + ntopics[0] - 1
                    opt_out_df_out = pd.concat([noise_df, opt_out_df_nn], ignore_index=True)
                    summary_out = pd.concat([noise_summary, opt_out_sum_nn], ignore_index=True)

                    # Write results
                    summary_out.to_csv(os.path.join(out_path,f"{ds}s_recluster_{recluster}_Income_{level}.csv"), index=False)
                    opt_out_df_out[["Author", "Year", "Income_level", ds , "Cluster", "ClusterName"]].to_csv(os.path.join(out_path,f"{ds}s_output_recluster_{recluster}_Income_{level}.csv"), index=False)
                    all_count = summary_out['Count'][summary_out['Cluster']>-1].sum()
                    noise_count = summary_out['Count'][summary_out['Cluster']==-1].values[0]
                    ICS_avg = summary_out['IntraClusterSimilarity'][summary_out['Cluster']>-1].mean()
                    newrow = {"Type":ds,"Income":level,"Count":all_count,"Noise":noise_count,"IC_Similarity":ICS_avg,"Recluster":recluster,"Quartile":qt,"Success":solution_rec.success,"Best_min_size":int(opt_mcs),"Best_threshold":opt_thr,"Best_min_size_inc":int(solution_rec.x[1]),"Best_threshold_inc":solution_rec.x[0]}
                    res.loc[len(res)] = newrow
                    print("\n Total time =",  time.time()-st_time)

print(res)
res.to_csv(os.path.join(out_path,f"Overall_results.csv"), index=False)