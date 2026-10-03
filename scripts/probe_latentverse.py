import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ml4h.explorations import stratify_and_project_latent_space, iterative_subspace_removal

def probe_model(latent_tsv: str, explore_csv: str, protected_attributes: list, output_dir: str):
    print(f"Probing Latentverse Model from {latent_tsv}...")
    
    # Load data
    df = pd.read_csv(explore_csv)
    df['sample_id'] = pd.to_numeric(df['fpath'], errors='coerce')
    df2 = pd.read_csv(latent_tsv, sep='\t', engine='python')
    df2['sample_id'] = pd.to_numeric(df2['sample_id'], errors='coerce')
    latent_df = pd.merge(df, df2, on='sample_id', how='inner')
    
    latent_cols = [c for c in latent_df.columns if 'latent' in c]
    if not latent_cols:
        raise ValueError("No columns with 'latent' found in the TSV file.")
    
    print(f"Found {len(latent_cols)} latent dimensions. Probing for: {protected_attributes}")
    
    all_scores = defaultdict(dict)
    
    for c in protected_attributes:
        if c not in latent_df.columns:
            print(f"Warning: {c} not found in dataset. Skipping.")
            continue
            
        # Drop NaNs for this probe
        valid_df = latent_df.dropna(subset=[c])
        if len(valid_df) < 10:
            print(f"Warning: Not enough valid data for {c}. Skipping.")
            continue
            
        is_categorical = valid_df[c].nunique() <= 2
        thresh = 1 if is_categorical else valid_df[c].median()
        
        scores = stratify_and_project_latent_space(c, thresh, 0, latent_cols, valid_df)
        all_scores['Raw Embeddings'].update(scores)
        
    print("\n--- Fairness Probe Results (Raw) ---")
    for attr, (tstat, pval, n) in all_scores['Raw Embeddings'].items():
        print(f"{attr}: T-Stat = {tstat:.3f}, P-Value = {pval:.3e} (n={n})")
        
    # Run Debiasing
    print("\nRunning iterative subspace removal to debias embeddings...")
    new_cols, debiased_df = iterative_subspace_removal(
        protected_attributes, latent_df, latent_cols, r2_thresh=0.01
    )
    
    print(f"Debiasing complete. New dimensions: {len(new_cols)}")
    
    # Probe again
    for c in protected_attributes:
        if c not in debiased_df.columns: continue
        valid_df = debiased_df.dropna(subset=[c])
        if len(valid_df) < 10: continue
        is_categorical = valid_df[c].nunique() <= 2
        thresh = 1 if is_categorical else valid_df[c].median()
        scores = stratify_and_project_latent_space(c, thresh, 0, new_cols, valid_df)
        all_scores['Debiased Embeddings'].update(scores)
        
    print("\n--- Fairness Probe Results (Debiased) ---")
    for attr, (tstat, pval, n) in all_scores['Debiased Embeddings'].items():
        print(f"{attr}: T-Stat = {tstat:.3f}, P-Value = {pval:.3e} (n={n})")
        
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_df = pd.DataFrame(all_scores).T
    report_df.to_csv(os.path.join(output_dir, 'fairness_report.csv'))
    print(f"\nReport saved to {output_dir}/fairness_report.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Latentverse Models for Fairness")
    parser.add_argument('--latent', required=True, help="Path to TSV containing latent embeddings")
    parser.add_argument('--metadata', required=True, help="Path to CSV containing protected attributes")
    parser.add_argument('--attrs', nargs='+', default=['age', 'sex', 'bmi'], help="Protected attributes to probe")
    parser.add_argument('--out', default='./fairness_results', help="Output directory")
    
    args = parser.parse_args()
    probe_model(args.latent, args.metadata, args.attrs, args.out)
