import numpy as np
import pandas as pd
from ml4h.explorations import confounder_matrix, iterative_subspace_removal

def test_confounder_matrix_handles_nans():
    # Synthetic latent space
    space = np.random.randn(100, 10)
    
    # Synthetic labels with NaNs
    df = pd.DataFrame({
        'Age': np.random.randn(100),
        'Sex': np.random.choice([0, 1], size=100)
    })
    df.loc[10:20, 'Age'] = np.nan
    
    cfm, scores = confounder_matrix(['Age', 'Sex'], df, space)
    assert cfm.shape == (2, 10), "Should handle NaNs gracefully and return vectors for both columns"
    assert 'Age' in scores and 'Sex' in scores

def test_iterative_subspace_removal_reduces_correlation():
    # Synthetic latent space (100 samples, 5 dims)
    space = np.random.randn(100, 5)
    
    # Create a protected attribute 'Age' that is highly correlated with the first dimension
    age = space[:, 0] * 5.0 + np.random.randn(100) * 0.1
    
    # Create latent dataframe
    latent_cols = [f'latent_{i}' for i in range(5)]
    latent_df = pd.DataFrame(space, columns=latent_cols)
    latent_df['Age'] = age
    
    # Check initial correlation
    _, initial_scores = confounder_matrix(['Age'], latent_df, space)
    assert initial_scores['Age'] > 0.9, "Initial correlation should be high"
    
    # Run removal
    new_cols, new_df = iterative_subspace_removal(['Age'], latent_df, latent_cols, r2_thresh=0.01)
    
    # Check debiased correlation
    new_space = new_df[new_cols].to_numpy()
    _, debiased_scores = confounder_matrix(['Age'], new_df, new_space)
    
    assert 'Age' not in debiased_scores or debiased_scores['Age'] < 0.01, "Debiased correlation should be < 0.01"
