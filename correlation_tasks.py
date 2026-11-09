from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    
    male_df = df[df["Gender"] == "Male"]
    female_df = df[df["Gender"] == "Female"]
    
    men_count = len(male_df)
    women_count = len(female_df)
    
    men_mri_correlation = {
        feat: float(male_df[feat].corr(male_df["MRI_Count"], method="pearson"))
        for feat in features
    }
    
    women_mri_correlation = {
        feat: float(female_df[feat].corr(female_df["MRI_Count"], method="pearson"))
        for feat in features
    }
    
    max_corr = -1.0
    strongest_mri_feature = ""
    
    for feat in features:
        m_abs = abs(men_mri_correlation[feat])
        if m_abs > max_corr:
            max_corr = m_abs
            strongest_mri_feature = feat
            
        f_abs = abs(women_mri_correlation[feat])
        if f_abs > max_corr:
            max_corr = f_abs
            strongest_mri_feature = feat
            
    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature
    )
