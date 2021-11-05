import os
import multiprocessing
from collections import defaultdict

import pysam
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Ridge

from scipy.stats import pearsonr
LATENT_COLS=[f'new_latent_24_{i}' for i in range(86)]
ADJUST_COLS=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11',
               'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21',
               'gt_array_axiom', 'gt_batch', 'assessment_center', 'age', 'age_squared', 'sex']

def run():
    input_bcf = os.environ['INPUT_BCF']
    latent_df = os.environ['INPUT_LATENT']
    output_csv = os.environ['OUTPUT_CSV']
    chrom = os.environ['CHROM']
    start = os.environ['START']
    stop = os.environ['STOP']
    latent_space_gwas(input_bcf, chrom, start, stop, latent_df, LATENT_COLS, ADJUST_COLS, output_csv)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            angle_between((1, 0, 0), (0, 1, 0))
            90
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            180
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / 3.141592


def optimize_genotype_vector(label_header, df, indexes, verbose=False):
    clf = Ridge(normalize=True, max_iter=10000)
    # ElasticNet(max_iter=50000, normalize=True) #LinearRegression() #LinearRegression(normalize=True)   #ElasticNet(normalize=True)
    clf.fit(df[indexes], df[label_header])
    train_score = clf.score(df[indexes], df[label_header])
    if verbose:
        print(f'{label_header} Train R^2:{train_score:.3f}\n')
    return clf.coef_


def get_genotype_vector_and_angle(snp, latent_cols, latent_df):
    ref = latent_df.loc[latent_df[snp] == 0][latent_cols].to_numpy()
    het = latent_df.loc[latent_df[snp] == 1][latent_cols].to_numpy()
    hom = latent_df.loc[latent_df[snp] == 2][latent_cols].to_numpy()

    ref_centroid = np.mean(ref, axis=0)
    het_centroid = np.mean(het, axis=0)
    hom_centroid = np.mean(hom, axis=0)

    ref2het = het_centroid - ref_centroid
    ref2hom = hom_centroid - ref_centroid
    het2hom = hom_centroid - het_centroid

    angle = angle_between(ref2het, het2hom)
    snp_vector = (ref2het + ref2hom) / 2

    het_weight = het.shape[0] / (het.shape[0] + hom.shape[0])
    hom_weight = hom.shape[0] / (het.shape[0] + hom.shape[0])
    # print(f'Weights are {het_weight:0.3f} {hom_weight:0.3f}  from {het.shape[0]} and {hom.shape[0]}')
    snp_vector = (het_weight * ref2het) + (hom_weight * ref2hom)

    return snp_vector, angle


def confounder_vector(label_header, df, indexes):
    clf = Ridge(normalize=True,
                max_iter=50000)  # ElasticNet(max_iter=50000, normalize=True)#Ridge(normalize=True, max_iter=10000)#ElasticNet(max_iter=50000, normalize=True) #LinearRegression() #LinearRegression(normalize=True)   #ElasticNet(normalize=True)
    clf.fit(df[indexes], df[label_header])
    train_score = clf.score(df[indexes], df[label_header])
    return clf.coef_, train_score


def confounder_matrix(adjust_cols, df, indexes):
    vectors = []
    scores = {}
    for col in adjust_cols:
        cv, r2 = confounder_vector(col, df, indexes)
        scores[col] = r2
        vectors.append(cv)
    return np.array(vectors), scores


def iterative_subspace_removal(adjust_cols, latent_df, latent_cols, r2_thresh=0.01):
    new_cols = latent_cols
    new_adjust_cols = adjust_cols
    space = latent_df[latent_cols].to_numpy()
    iteration = 0
    while len(new_adjust_cols) > 0 and space.shape[-1] > len(new_adjust_cols):
        cfm, scores = confounder_matrix(new_adjust_cols, latent_df, new_cols)
        u, s, vt = np.linalg.svd(cfm, full_matrices=True)
        nspace = np.matmul(space, vt[:, len(new_adjust_cols):])
        new_cols = []
        for i in range(nspace.shape[-1]):
            col = f'new_latent_{iteration}_{i}'
            new_cols.append(col)
            latent_df[col] = nspace[:, i]

        iteration += 1
        space = nspace
        new_adjust_cols = [col for col, score in scores.items() if score > r2_thresh]
        print(f'Scores were {scores}, remaining columns are {new_adjust_cols}')
        print(f'After iteration {iteration} Space shape is: {space.shape}')
    return new_cols


def stratify_genotype_and_project_latent_space(stratify_column, latent_cols, latent_df, adjust_cols,
                                               manova=True, optimize=False):
    latent_df = latent_df[[stratify_column] + latent_cols + adjust_cols].dropna()
    genotype_vector, angle = get_genotype_vector_and_angle(stratify_column, latent_cols, latent_df)
    if manova:
        formula = f"{'+'.join(latent_cols)} ~ {stratify_column}"
        maov = MANOVA.from_formula(formula, data=latent_df)
        test = maov.mv_test()
        s = test[stratify_column]['stat']
        return s['F Value'][0], s['Pr > F'][0], s['Value'][0], s['Value'][0], angle, genotype_vector

    if optimize:
        genotype_vector = optimize_genotype_vector(stratify_column, latent_df, latent_cols, verbose=True)
    space = latent_df[latent_cols].to_numpy()
    all_dots = np.array([np.dot(genotype_vector, v) for v in space])
    all_genotypes = latent_df[stratify_column].to_numpy()

    if len(adjust_cols) > 0:
        all_adjustments = latent_df[adjust_cols].to_numpy()
        all_data = np.column_stack([all_genotypes, all_adjustments, np.ones(all_dots.shape[0])])
        clean_cols = [col.replace('22009', '').replace('21003', '').replace('-', '').replace('_', '') for col in adjust_cols]
        formula = f'y ~ genotypes + {" + ".join(clean_cols)}'
    else:
        all_data = np.column_stack([all_genotypes, np.ones(all_dots.shape[0])])
        formula = f'y ~ genotypes'
        clean_cols = adjust_cols
    data = {'y': all_dots, 'genotypes': all_genotypes}
    for i, col in enumerate(clean_cols):
        data[col] = all_adjustments[:, i]
    df = pd.DataFrame.from_dict(data)

    results = smf.ols(formula, data=df).fit()
    p_value = float(results.summary2().tables[1]['P>|t|']['genotypes'])
    t_stat = float(results.summary2().tables[1]['t']['genotypes'])
    coef = float(results.summary2().tables[1]['Coef.']['genotypes'])
    se = float(results.summary2().tables[1]['Std.Err.']['genotypes'])
    return t_stat, p_value, coef, se, angle, genotype_vector


def merge_snp(latent_df, snp_vcf, snp_id):
    genos = {'1,0,0': 0, '0,1,0': 1, '0,0,1': 2}
    with open(snp_vcf, mode='r') as vcf:
        vcf.readline()
        vcf.readline()
        huge_line = vcf.readline()
        splits = huge_line.split()
        huge_line2 = vcf.readline()
        splits2 = huge_line2.split()
        sample2genos = {}
        for s, g in zip(splits[9:], splits2[9:]):
            if g in genos:
                sample2genos[int(s)] = genos[g]
            else:
                sample2genos[int(s)] = np.argmax(np.array(g))
    data = {'sample_id': list(sample2genos.keys()), snp_id: list(sample2genos.values())}
    genos = pd.DataFrame.from_dict(data)
    new_df = pd.merge(latent_df, genos, left_on='fpath', right_on='sample_id', how='inner')
    print(f' value counts: {snp_id} is : {new_df[snp_id].value_counts()}')
    return new_df


def merge_snps(latent_df):
    latent_df = merge_snp(latent_df, '/home/sam/extract_rs883079.vcf', 'rs883079')  # TBX5 PRInterval ECG
    return latent_df


def latent_space_dataframe(infer_hidden_tsv, explore_csv):
    df = pd.read_csv(explore_csv, sep='\t')
    df['FID'] = pd.to_numeric(df['FID'], errors='coerce')
    df2 = pd.read_csv(infer_hidden_tsv, sep='\t')
    df2['sample_id'] = pd.to_numeric(df2['sample_id'], errors='coerce')
    latent_df = pd.merge(df, df2, left_on='FID', right_on='sample_id', how='inner')
    latent_df.info()
    return latent_df


def latent_space_gwas(input_bcf, chrom, start, stop, latent_df, latent_cols, adjust_cols, output_file,
                      optimize_genotype_vector=False):
    remap = [1, 0]
    gv_dict = defaultdict(list)
    #bcf_in = pysam.VariantFile(f"/mnt/disks/chr12-bcfs/latent_space_ukb_imp_chr{chrom}_v3_geno_095_maf_0005_info_03.bcf.bcf")
    bcf_in = pysam.VariantFile(input_bcf)
    for i, rec in enumerate(bcf_in.fetch(chrom, int(start), int(stop))):
        sample2genos = {}
        for j, s in enumerate(rec.samples.values()):
            g = s.values()[0]
            if g[0] is None or g[1] is None:
                continue
            sample_id = int(s.name.split("_")[0])
            sample2genos[sample_id] = remap[g[0]] + remap[g[1]]
        if sum(sample2genos.values()) > 1000:
            snp_id = f"snp_{rec.id.replace(':', '_')}"
            data = {'sample_id': list(sample2genos.keys()), snp_id: list(sample2genos.values())}
            genos = pd.DataFrame.from_dict(data)
            new_df = pd.merge(latent_df, genos, left_on='sample_id', right_on='sample_id', how='inner')
            counts = new_df[snp_id].value_counts()

            if len(counts) == 3 and counts[0] > 200:
                t_stat, p_value, coef, se, angle, gv = stratify_genotype_and_project_latent_space(snp_id, latent_cols,
                                                                                                  new_df, adjust_cols,
                                                                                                  optimize=optimize_genotype_vector)

                if len(gv_dict['rsid']) % 10 == 0:
                    print(f'Processed SNPs {len(gv_dict["rsid"])}, P_value: {p_value:0.4E}, pos: {rec.pos}')

                gv_dict['t_stat'].append(t_stat)
                gv_dict['p_value'].append(p_value)
                gv_dict['log10p'].append(-np.log10(p_value))
                gv_dict['coef'].append(coef)
                gv_dict['se'].append(se)
                gv_dict['angle'].append(angle)
                gv_dict['rsid'].append(rec.id)
                gv_dict['pos'].append(rec.pos)
                gv_dict['chrom'].append(rec.chrom)
                gv_dict['ref'].append(rec.ref)
                gv_dict['allele1'].append(rec.alleles[1])
                a1freq = ((0.5 * counts[1]) + counts[2]) / (counts[0] + counts[1] + counts[2])
                gv_dict['a1freq'].append(a1freq)
                gv_dict['ref_count'].append(counts[0])
                gv_dict['het_count'].append(counts[1])
                gv_dict['hom_count'].append(counts[2])
                for ii, v in enumerate(gv):
                    gv_dict[f'gv_{ii}'].append(v)

    print(f'Finished with total SNPs:{len(gv_dict["rsid"])}. Now write CSV.')
    gv_df = pd.DataFrame.from_dict(gv_dict)
    gv_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    run()  # back to the top