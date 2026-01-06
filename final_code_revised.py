import pandas as pd
import numpy as np
import ast
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings

# ======================================================================
# CONFIGURATION & HELPER FUNCTIONS
# ======================================================================
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def safe_parse_list(val):
    """Data Mining Preprocessing: Safely parses stringified lists."""
    if pd.isna(val) or val in ['none', 'nan', 'null', '']: return []
    val = str(val).strip()
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(s).strip() for s in parsed if s]
    except (ValueError, SyntaxError):
        pass
    return [s.strip().strip("'\"") for s in val.split(',') if s.strip()]

def parse_experience(val):
    """Data Cleaning: Extracts numeric experience years from noisy strings."""
    if pd.isna(val): return np.nan
    s = str(val).lower().strip()
    if s in ['fresher', 'freshers', 'entry level', 'none']: return 0.0
    match = re.search(r'(\d+(?:\.\d+)?)', s)
    return float(match.group(1)) if match else np.nan

def save_plot(filename):
    """Visualization Helper"""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {filename}")
    plt.show()

# ======================================================================
# DATA LOADING & PREPROCESSING STEP
# ======================================================================
def load_and_process_data(filepath):
    print("="*60 + "\nDATA MINING PIPELINE: LOADING & PREPROCESSING\n" + "="*60)
    if not os.path.exists(filepath):
        print(f"✗ Error: '{filepath}' not found."); return None

    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows.")

    # 1. Cleaning & Feature Extraction
    df['candidate_skills_clean'] = df['Skills'].apply(safe_parse_list)
    df['required_skills_clean'] = df['skills_required'].apply(safe_parse_list)
    
    df['experience_years'] = df['experiencere_requirement'].apply(parse_experience)
    df['experience_years'].fillna(df['experience_years'].median(), inplace=True)
    
    df['job_role'] = df['job_position_name']
    df['education_clean'] = df['educationaL_requirements'].astype(str).replace({'nan': None})
    
    # 2. Feature Engineering (for Correlation Mining)
    df['n_cand_skills'] = df['candidate_skills_clean'].apply(len)
    
    # Jaccard Similarity concept for skill matching
    df['skill_match_ratio'] = df.apply(lambda r: len(set(r['candidate_skills_clean']) & set(r['required_skills_clean'])) / len(r['required_skills_clean']) if r['required_skills_clean'] else 0, axis=1)
    
    if 'matched_score' in df.columns:
        df['score'] = pd.to_numeric(df['matched_score'], errors='coerce')
    
    return df

# ======================================================================
# QUESTION 1: Which skill is required for which role?
# Concept: Association Analysis (Role -> Skills)
# ======================================================================
def analyze_roles_and_skills(df):
    print("\n" + "="*60)
    print("QUESTION 1: Which skill is required for which role?")
    print("="*60)
    
    # Aggregation
    role_skills = {}
    for _, row in df.iterrows():
        if row['job_role'] and row['required_skills_clean']:
            role_skills.setdefault(row['job_role'], []).extend(row['required_skills_clean'])

    # Visualization of Top Roles
    top_roles = sorted(role_skills.keys(), key=lambda k: len(role_skills[k]), reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Top Skills by Role (Frequent Itemsets)', fontsize=16, fontweight='bold')
    
    for idx, role in enumerate(top_roles):
        ax = axes[idx//3, idx%3]
        common = Counter(role_skills[role]).most_common(5)
        if common:
            skills, counts = zip(*common)
            ax.barh(skills, counts, color=plt.cm.Set3(idx/6))
            ax.invert_yaxis()
            ax.set_title(role[:30], fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Data')
    
    save_plot('skills_by_role.png')
    return role_skills

# ======================================================================
# QUESTION 2: Which is the most in-demand skill?
# QUESTION 3: Which skill is most common among candidates?
# Concept: Frequency Analysis & Gap Analysis
# ======================================================================
def analyze_demand_supply(df):
    print("\n" + "="*60)
    print("QUESTION 2: Which is the most in-demand skill?")
    print("QUESTION 3: Which skill is most common among candidates?")
    print("="*60)
    
    # Flattening (Bag of Words concept)
    all_req = [s for sublist in df['required_skills_clean'] for s in sublist]
    all_cand = [s for sublist in df['candidate_skills_clean'] for s in sublist]
    
    req_counts = Counter(all_req)
    cand_counts = Counter(all_cand)
    
    # 1. Bar Chart (Demand)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    top_req = req_counts.most_common(10)
    ax1.barh([x[0] for x in top_req], [x[1] for x in top_req], color='skyblue')
    ax1.invert_yaxis()
    ax1.set_title("Top 10 In-Demand Skills")
    
    # 2. Pie Chart (Distribution)
    ax2.pie([x[1] for x in top_req[:5]], labels=[x[0] for x in top_req[:5]], autopct='%1.1f%%')
    ax2.set_title("Distribution of Top 5 Skills")
    save_plot('most_in_demand_skills.png')

    # 3. Supply-Demand Gap Analysis
    plt.figure(figsize=(12, 6))
    top_skills = [x[0] for x in req_counts.most_common(10)]
    x = np.arange(len(top_skills))
    width = 0.35
    plt.bar(x - width/2, [req_counts[s] for s in top_skills], width, label='Job Demand')
    plt.bar(x + width/2, [cand_counts[s] for s in top_skills], width, label='Candidate Supply')
    plt.xticks(x, top_skills, rotation=45)
    plt.legend()
    plt.title("Demand vs Supply Gap for Top Skills")
    save_plot('most_common_skills_candidates.png')
    
    # 4. Text Mining: Word Cloud
    print("\nGenerating Skills Word Cloud...")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_req))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    save_plot('skills_wordcloud.png')

    return req_counts

# ======================================================================
# QUESTION 4: What is the educational requirement for a role?
# QUESTION 5: What is the experience requirement for a job role?
# Concept: Descriptive Statistics & Distribution Analysis
# ======================================================================
def analyze_education_experience(df):
    print("\n" + "="*60)
    print("QUESTION 4: What is the educational requirement for a role?")
    print("QUESTION 5: What is the experience requirement for a job role?")
    print("="*60)
    
    # Education Analysis
    edu_counts = df['education_clean'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    edu_counts.plot(kind='barh', color='lightgreen')
    plt.title('Top Educational Requirements')
    plt.gca().invert_yaxis()
    save_plot('education_requirements.png')

    # Experience Analysis (Distribution)
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(df['experience_years'].dropna(), bins=15, color='darkgreen', alpha=0.7)
    plt.title('Experience Distribution (Histogram)')
    plt.xlabel('Years')
    
    # Boxplot (Outlier Detection)
    plt.subplot(1, 2, 2)
    top_roles = df['job_role'].value_counts().head(10).index
    data_to_plot = [df[df['job_role'] == r]['experience_years'].dropna() for r in top_roles]
    plt.boxplot(data_to_plot, labels=[r[:15] for r in top_roles])
    plt.xticks(rotation=45)
    plt.title('Experience by Top Roles (Boxplot)')
    
    save_plot('experience_distribution.png')

# ======================================================================
# QUESTION 6: Are people with higher experience and skills preferred?
# Concept: Correlation Analysis & Relationship Mining
# ======================================================================
def analyze_preferences(df):
    print("\n" + "="*60)
    print("QUESTION 6: Are people with higher experience and skills preferred?")
    print("="*60)
    
    if 'score' not in df.columns or df['score'].notna().sum() < 10:
        print("Insufficient data for correlation analysis.")
        return

    # Correlation Matrix
    subset = df[['score', 'experience_years', 'n_cand_skills', 'skill_match_ratio']].dropna()
    corr = subset.corr()
    print("Correlation Matrix:\n", corr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Regression Plot
    sns.regplot(data=subset, x='experience_years', y='score', ax=axes[0], scatter_kws={'alpha':0.5})
    axes[0].set_title('Experience vs Match Score (Regression)')
    
    # Heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1])
    axes[1].set_title('Feature Correlation Heatmap')
    
    save_plot('preference_analysis.png')

# ======================================================================
# MAIN EXECUTION
# ======================================================================
def print_summary(df, req_counts):
    print("\n" + "="*60 + "\nMINING RESULTS SUMMARY\n" + "="*60)
    print(f"1. Top Skill (Frequent Item): {req_counts.most_common(1)[0][0]}")
    print(f"2. Avg Experience (Mean): {df['experience_years'].mean():.1f} years")
    print(f"3. Top Education (Mode): {df['education_clean'].mode()[0]}")
    print(f"4. Dataset Size: {len(df)} records")
    print("✓ Data Mining Complete. All patterns visualized.")

if __name__ == "__main__":
    df = load_and_process_data('resume_data.csv')
    
    if df is not None:
        analyze_roles_and_skills(df)
        req_counts = analyze_demand_supply(df)
        analyze_education_experience(df)
        analyze_preferences(df)
        print_summary(df, req_counts)