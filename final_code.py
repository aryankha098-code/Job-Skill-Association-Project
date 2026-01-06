import pandas as pd
import numpy as np
import ast
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ======================================================================
# LOAD AND INSPECT DATA
# ======================================================================
print("Loading data...")
try:
    df = pd.read_csv('resume_data.csv')
    print(f"✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns in dataset: {', '.join(df.columns.tolist())}")
except FileNotFoundError:
    print("✗ Error: 'resume_data.csv' file not found.")
    print("  Please ensure the file is in the current directory.")
    exit()
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit()

# Show basic info
print(f"\nFirst few rows:")
print(df.head())
print(f"\nMissing values per column:")
print(df.isnull().sum())

# ======================================================================
# DATA PREPROCESSING
# ======================================================================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

def extract_skills(skill_string):
    """Extract skills from string representation of list"""
    if pd.isna(skill_string) or skill_string is None:
        return []
    
    try:
        # Handle different formats
        if isinstance(skill_string, str):
            # Clean the string
            skill_string = str(skill_string).strip()
            
            # Handle empty strings
            if not skill_string or skill_string.lower() in ['none', 'nan', 'null', '']:
                return []
            
            # Try to parse as list if it looks like one
            if skill_string.startswith('[') and skill_string.endswith(']'):
                try:
                    # Safely evaluate the string as a list
                    skills_list = ast.literal_eval(skill_string)
                    if isinstance(skills_list, list):
                        # Clean each skill
                        cleaned_skills = []
                        for skill in skills_list:
                            if skill and str(skill).strip().lower() not in ['none', 'nan', 'null']:
                                cleaned_skills.append(str(skill).strip())
                        return cleaned_skills
                except (ValueError, SyntaxError):
                    pass
            
            # If not a list, split by comma
            skills = [s.strip().strip('"').strip("'").strip() 
                     for s in skill_string.split(',') if s.strip()]
            return [s for s in skills if s and s.lower() not in ['none', 'nan', 'null']]
        return []
    except Exception as e:
        print(f"Warning: Error parsing skills: {skill_string[:50]}... Error: {e}")
        return []

# Create cleaned skills columns
print("\nCleaning skills data...")
df['candidate_skills_clean'] = df['Skills'].apply(extract_skills)
df['required_skills_clean'] = df['skills_required'].apply(extract_skills)

# Create job role column
df['job_role'] = df['job_position_name']

# Create experience column (extract numeric values)
def extract_experience(exp_string):
    """Extract numeric experience from string"""
    if pd.isna(exp_string) or exp_string is None:
        return np.nan
    
    try:
        exp_string = str(exp_string).strip().lower()
        
        # Check for common non-numeric values
        if exp_string in ['none', 'nan', 'null', 'fresher', 'freshers', 'entry level']:
            return 0
        
        # Extract numbers using regex
        numbers = re.findall(r'(\d+(?:\.\d+)?)', exp_string)
        if numbers:
            # Take the first number found
            return float(numbers[0])
        
        # Try to extract from common patterns
        patterns = [
            r'(\d+)\s*(?:years?|yrs?|y\.?)',
            r'(\d+)\+',
            r'experience.*?(\d+)',
            r'(\d+)\s*to\s*\d+',
            r'(\d+)\s*-\s*\d+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, exp_string, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return np.nan
    except Exception as e:
        print(f"Warning: Error parsing experience: {exp_string[:50]}... Error: {e}")
        return np.nan

print("Extracting experience data...")
df['experience_years'] = df['experiencere_requirement'].apply(extract_experience)

# Fill NaN experience with median or 0
exp_median = df['experience_years'].median()
if pd.isna(exp_median):
    exp_median = 0
df['experience_years'] = df['experience_years'].fillna(exp_median)

# Create education requirement column
df['education_requirement'] = df['educationaL_requirements']

print("✓ Data preprocessing completed!")
print(f"  Records with candidate skills: {df['candidate_skills_clean'].apply(lambda x: len(x) > 0).sum()}")
print(f"  Records with required skills: {df['required_skills_clean'].apply(lambda x: len(x) > 0).sum()}")
print(f"  Average experience: {df['experience_years'].mean():.1f} years")

# ======================================================================
# QUESTION 1: Which skill is required for which role?
# ======================================================================
print("\n" + "="*60)
print("QUESTION 1: Which skill is required for which role?")
print("="*60)

# Flatten skills for each role
role_skills = {}
for idx, row in df.iterrows():
    role = row['job_role']
    skills = row['required_skills_clean']
    
    if pd.isna(role) or not skills or not isinstance(skills, list):
        continue
    
    if role not in role_skills:
        role_skills[role] = []
    role_skills[role].extend(skills)

# Show top skills for each role
print(f"\nFound {len(role_skills)} unique job roles")
print("\nTop skills required for each role (first 5 roles):")
print("-" * 60)

if role_skills:
    roles_list = list(role_skills.items())[:5]
    for role, skills in roles_list:
        if skills:
            skill_counter = Counter(skills)
            top_5 = skill_counter.most_common(5)
            print(f"\n{role}:")
            for skill, count in top_5:
                print(f"  • {skill}: {count} times")
        else:
            print(f"\n{role}: No skills specified")
else:
    print("No role-skill data available.")

# Visualization for top roles
if role_skills:
    top_roles = list(role_skills.keys())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Top Skills Required by Role', fontsize=16, fontweight='bold')
    
    for idx, role in enumerate(top_roles):
        skills = role_skills[role]
        skill_counter = Counter(skills)
        top_skills = skill_counter.most_common(5)
        
        if top_skills:
            ax = axes[idx // 3, idx % 3]
            skills_list, counts = zip(*top_skills)
            y_pos = np.arange(len(skills_list))
            
            bars = ax.barh(y_pos, counts, color=plt.cm.Set3(idx/6))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(skills_list, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Frequency', fontsize=10)
            ax.set_title(f'{role}', fontsize=11, fontweight='bold')
            
            # Add count labels
            for i, v in enumerate(counts):
                ax.text(v + 0.5, i, str(v), va='center', fontsize=9)
        else:
            axes[idx // 3, idx % 3].text(0.5, 0.5, 'No data', 
                                       ha='center', va='center', fontsize=12)
            axes[idx // 3, idx % 3].set_title(f'{role}', fontsize=11)
    
    # Remove empty subplots
    for idx in range(len(top_roles), 6):
        axes[idx // 3, idx % 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('skills_by_role.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No data available for visualization.")

# ======================================================================
# QUESTION 2: Which is the most in-demand skill?
# ======================================================================
print("\n" + "="*60)
print("QUESTION 2: Which is the most in-demand skill?")
print("="*60)

# Count all required skills
all_required_skills = []
for skills in df['required_skills_clean'].dropna():
    if isinstance(skills, list):
        all_required_skills.extend(skills)

if all_required_skills:
    skill_counter = Counter(all_required_skills)
    top_20_demand_skills = skill_counter.most_common(20)
    
    print(f"\nFound {len(skill_counter)} unique skills")
    print("Top 20 Most In-Demand Skills:")
    print("-" * 50)
    
    for i, (skill, count) in enumerate(top_20_demand_skills, 1):
        percentage = (count / len(df)) * 100
        print(f"{i:2}. {skill:30} - {count:3} occurrences ({percentage:.1f}% of jobs)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart for top 10
    top_10 = skill_counter.most_common(10)
    skills, counts = zip(*top_10)
    bars = ax1.barh(range(len(skills)), counts, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(skills))))
    ax1.set_yticks(range(len(skills)))
    ax1.set_yticklabels(skills)
    ax1.invert_yaxis()
    ax1.set_xlabel('Number of Job Postings')
    ax1.set_title('Top 10 Most In-Demand Skills', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{count}', va='center', fontweight='bold')
    
    # Pie chart for top 5
    top_5 = skill_counter.most_common(5)
    if top_5:
        skills_pie, counts_pie = zip(*top_5)
        colors = plt.cm.Set3(np.arange(len(skills_pie)))
        wedges, texts, autotexts = ax2.pie(counts_pie, labels=skills_pie, autopct='%1.1f%%', 
                                          colors=colors, startangle=90, textprops={'fontsize': 9})
        ax2.set_title('Top 5 Skills Distribution', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
        ax2.set_title('Top 5 Skills Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('most_in_demand_skills.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No required skills data available.")

# ======================================================================
# QUESTION 3: Which skill is most common among candidates?
# ======================================================================
print("\n" + "="*60)
print("QUESTION 3: Which skill is most common among candidates?")
print("="*60)

# Count all candidate skills
all_candidate_skills = []
for skills in df['candidate_skills_clean'].dropna():
    if isinstance(skills, list):
        all_candidate_skills.extend(skills)

if all_candidate_skills:
    candidate_skill_counter = Counter(all_candidate_skills)
    top_20_common_skills = candidate_skill_counter.most_common(20)
    
    print(f"\nFound {len(candidate_skill_counter)} unique candidate skills")
    print("Top 20 Most Common Skills Among Candidates:")
    print("-" * 55)
    
    for i, (skill, count) in enumerate(top_20_common_skills, 1):
        percentage = (count / len(df)) * 100
        print(f"{i:2}. {skill:30} - {count:3} candidates ({percentage:.1f}% of candidates)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart for top 10
    top_10_common = candidate_skill_counter.most_common(10)
    common_skills, common_counts = zip(*top_10_common)
    bars = ax1.barh(range(len(common_skills)), common_counts, 
                    color=plt.cm.Oranges(np.linspace(0.4, 0.8, len(common_skills))))
    ax1.set_yticks(range(len(common_skills)))
    ax1.set_yticklabels(common_skills)
    ax1.invert_yaxis()
    ax1.set_xlabel('Number of Candidates')
    ax1.set_title('Top 10 Most Common Skills Among Candidates', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, common_counts)):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{count}', va='center', fontweight='bold')
    
    # Demand vs Supply comparison
    if 'skill_counter' in locals() and skill_counter:
        # Get top skills from both
        demand_top = dict(skill_counter.most_common(10))
        supply_top = dict(candidate_skill_counter.most_common(10))
        
        all_top_skills = sorted(set(list(demand_top.keys()) + list(supply_top.keys())))
        
        demand_values = [demand_top.get(skill, 0) for skill in all_top_skills]
        supply_values = [supply_top.get(skill, 0) for skill in all_top_skills]
        
        x = np.arange(len(all_top_skills))
        width = 0.35
        
        ax2.bar(x - width/2, demand_values, width, label='Job Demand', color='steelblue', alpha=0.8)
        ax2.bar(x + width/2, supply_values, width, label='Candidate Supply', color='darkorange', alpha=0.8)
        ax2.set_xlabel('Skills')
        ax2.set_ylabel('Count')
        ax2.set_title('Demand vs Supply for Top Skills', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_top_skills, rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No demand data available\nfor comparison', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Demand vs Supply Analysis', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('most_common_skills_candidates.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No candidate skills data available.")

# ======================================================================
# QUESTION 4: What is the educational requirement for a role?
# ======================================================================
print("\n" + "="*60)
print("QUESTION 4: What is the educational requirement for a role?")
print("="*60)

# Analyze educational requirements
education_requirements = {}
for idx, row in df.iterrows():
    role = row['job_role']
    education = row['education_requirement']
    
    if pd.isna(role) or pd.isna(education):
        continue
    
    if role not in education_requirements:
        education_requirements[role] = []
    
    # Clean education text
    education = str(education).strip()
    if education and education.lower() not in ['none', 'nan', 'null', '']:
        education_requirements[role].append(education)

if education_requirements:
    print(f"\nFound education requirements for {len(education_requirements)} roles")
    print("Educational requirements by role (first 5 roles):")
    print("-" * 60)
    
    for role, reqs in list(education_requirements.items())[:5]:
        if reqs:
            counter = Counter(reqs)
            most_common = counter.most_common(3)
            print(f"\n{role}:")
            for req, count in most_common:
                print(f"  • {req} ({count} postings)")
        else:
            print(f"\n{role}: No education requirements specified")
    
    # Overall education requirements analysis
    all_education = []
    for reqs in education_requirements.values():
        all_education.extend(reqs)
    
    if all_education:
        education_counter = Counter(all_education)
        top_education = education_counter.most_common(10)
        
        # Visualization
        plt.figure(figsize=(12, 8))
        education_types, education_counts = zip(*top_education)
        
        bars = plt.barh(range(len(education_types)), education_counts, 
                        color=plt.cm.Greens(np.linspace(0.3, 0.8, len(education_types))))
        plt.yticks(range(len(education_types)), education_types)
        plt.gca().invert_yaxis()
        plt.xlabel('Number of Job Postings')
        plt.title('Top Educational Requirements Across All Roles', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, education_counts)):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{count}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('education_requirements.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("\nNo education requirement data available for visualization.")
else:
    print("No education requirement data available.")

# ======================================================================
# QUESTION 5: What is the experience requirement for a job role?
# ======================================================================
print("\n" + "="*60)
print("QUESTION 5: What is the experience requirement for a job role?")
print("="*60)

# Analyze experience requirements by role
experience_by_role = {}
for idx, row in df.iterrows():
    role = row['job_role']
    exp_years = row['experience_years']
    
    if pd.isna(role) or pd.isna(exp_years):
        continue
    
    if role not in experience_by_role:
        experience_by_role[role] = []
    experience_by_role[role].append(exp_years)

if experience_by_role:
    print(f"\nFound experience data for {len(experience_by_role)} roles")
    print("Experience requirements by role (first 5 roles):")
    print("-" * 60)
    
    for role, exps in list(experience_by_role.items())[:5]:
        if exps:
            avg_exp = np.mean(exps)
            min_exp = min(exps)
            max_exp = max(exps)
            std_exp = np.std(exps) if len(exps) > 1 else 0
            print(f"\n{role}:")
            print(f"  • Average: {avg_exp:.1f} years")
            print(f"  • Range: {min_exp} - {max_exp} years")
            print(f"  • Standard Deviation: {std_exp:.1f} years")
            print(f"  • Sample size: {len(exps)} postings")
        else:
            print(f"\n{role}: No experience data")
    
    # Overall experience distribution
    plt.figure(figsize=(14, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    exp_data = df['experience_years'].dropna()
    
    if not exp_data.empty:
        n, bins, patches = plt.hist(exp_data, bins=15, edgecolor='black', alpha=0.7, color='darkgreen')
        plt.xlabel('Years of Experience Required')
        plt.ylabel('Number of Job Postings')
        plt.title('Distribution of Experience Requirements', fontweight='bold')
        plt.axvline(exp_data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {exp_data.mean():.1f} years')
        plt.axvline(exp_data.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {exp_data.median():.1f} years')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No experience data available', 
                ha='center', va='center', fontsize=12)
        plt.title('Distribution of Experience Requirements', fontweight='bold')
    
    # Box plot by role (top roles with enough data)
    plt.subplot(1, 2, 2)
    top_roles_exp = []
    
    for role, exps in experience_by_role.items():
        if len(exps) >= 3:  # Only include roles with at least 3 postings
            avg_exp = np.mean(exps)
            top_roles_exp.append((role, avg_exp, len(exps)))
    
    if top_roles_exp:
        # Sort by average experience and take top 10
        top_roles_exp.sort(key=lambda x: x[1], reverse=True)
        top_10_roles_exp = top_roles_exp[:10]
        
        # Prepare data for box plot
        boxplot_data = []
        labels = []
        for role, avg_exp, count in top_10_roles_exp:
            boxplot_data.append(experience_by_role[role])
            labels.append(f"{role[:20]}...\n(n={count})")
        
        plt.boxplot(boxplot_data, labels=labels)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.ylabel('Years of Experience')
        plt.title('Experience Requirements by Role (Top 10)', fontweight='bold')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor box plot', 
                ha='center', va='center', fontsize=12)
        plt.title('Experience Requirements by Role', fontweight='bold')
    
    plt.savefig('experience_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No experience data available.")

# ======================================================================
# QUESTION 6: Are people with higher experience and skills preferred?
# ======================================================================
print("\n" + "="*60)
print("QUESTION 6: Are people with higher experience and skills preferred?")
print("="*60)

# Create features for analysis
df['num_candidate_skills'] = df['candidate_skills_clean'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['num_required_skills'] = df['required_skills_clean'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Calculate skill match ratio
def calculate_skill_match(row):
    try:
        if isinstance(row['candidate_skills_clean'], list) and isinstance(row['required_skills_clean'], list):
            candidate_set = set(row['candidate_skills_clean'])
            required_set = set(row['required_skills_clean'])
            if len(required_set) > 0:
                return len(candidate_set.intersection(required_set)) / len(required_set)
    except:
        pass
    return 0

df['skill_match_ratio'] = df.apply(calculate_skill_match, axis=1)

# Check if matched_score exists
if 'matched_score' in df.columns:
    print("\nAnalyzing candidate preferences based on match scores...")
    
    # Clean matched_score
    df['matched_score_clean'] = pd.to_numeric(df['matched_score'], errors='coerce')
    
    # Check if we have enough data
    valid_scores = df['matched_score_clean'].notna().sum()
    if valid_scores > 10:  # Need at least 10 valid scores for meaningful analysis
        print(f"Found {valid_scores} valid match scores for analysis")
        
        # Calculate correlations with proper data alignment
        corr_data = df[['matched_score_clean', 'experience_years', 'num_candidate_skills', 'skill_match_ratio']].dropna()
        
        if len(corr_data) > 1:
            correlations = corr_data.corr()
            
            print("\nCorrelation Analysis:")
            print("-" * 40)
            print("\nCorrelation Matrix:")
            print(correlations.round(3))
            
            # Visualize relationships
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Factors Affecting Candidate Preference', fontsize=16, fontweight='bold')
            
            # Experience vs Match Score
            axes[0, 0].scatter(df['experience_years'], df['matched_score_clean'], alpha=0.6, color='darkblue')
            axes[0, 0].set_xlabel('Experience (years)')
            axes[0, 0].set_ylabel('Match Score')
            axes[0, 0].set_title('Experience vs Match Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add trend line if enough data points
            temp_exp = df[['experience_years', 'matched_score_clean']].dropna()
            if len(temp_exp) > 2:
                try:
                    z = np.polyfit(temp_exp['experience_years'], temp_exp['matched_score_clean'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(temp_exp['experience_years'].min(), 
                                         temp_exp['experience_years'].max(), 100)
                    axes[0, 0].plot(x_range, p(x_range), "r--", alpha=0.8, label=f"y={z[0]:.3f}x+{z[1]:.3f}")
                    axes[0, 0].legend(fontsize=8)
                except:
                    pass
            
            # Number of Skills vs Match Score
            axes[0, 1].scatter(df['num_candidate_skills'], df['matched_score_clean'], alpha=0.6, color='darkorange')
            axes[0, 1].set_xlabel('Number of Candidate Skills')
            axes[0, 1].set_ylabel('Match Score')
            axes[0, 1].set_title('Skills Count vs Match Score')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add trend line if enough data points
            temp_skills = df[['num_candidate_skills', 'matched_score_clean']].dropna()
            if len(temp_skills) > 2:
                try:
                    z = np.polyfit(temp_skills['num_candidate_skills'], temp_skills['matched_score_clean'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(temp_skills['num_candidate_skills'].min(), 
                                         temp_skills['num_candidate_skills'].max(), 100)
                    axes[0, 1].plot(x_range, p(x_range), "r--", alpha=0.8, label=f"y={z[0]:.3f}x+{z[1]:.3f}")
                    axes[0, 1].legend(fontsize=8)
                except:
                    pass
            
            # Skill Match Ratio vs Match Score
            axes[1, 0].scatter(df['skill_match_ratio'], df['matched_score_clean'], alpha=0.6, color='darkgreen')
            axes[1, 0].set_xlabel('Skill Match Ratio')
            axes[1, 0].set_ylabel('Match Score')
            axes[1, 0].set_title('Skill Match Ratio vs Match Score')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add trend line if enough data points
            temp_match = df[['skill_match_ratio', 'matched_score_clean']].dropna()
            if len(temp_match) > 2:
                try:
                    z = np.polyfit(temp_match['skill_match_ratio'], temp_match['matched_score_clean'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(temp_match['skill_match_ratio'].min(), 
                                         temp_match['skill_match_ratio'].max(), 100)
                    axes[1, 0].plot(x_range, p(x_range), "r--", alpha=0.8, label=f"y={z[0]:.3f}x+{z[1]:.3f}")
                    axes[1, 0].legend(fontsize=8)
                except:
                    pass
            
            # Heatmap of correlations
            im = axes[1, 1].imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_xticks(range(len(correlations.columns)))
            axes[1, 1].set_xticklabels(correlations.columns, rotation=45, ha='right')
            axes[1, 1].set_yticks(range(len(correlations.columns)))
            axes[1, 1].set_yticklabels(correlations.columns)
            axes[1, 1].set_title('Correlation Heatmap')
            
            # Add correlation values to heatmap
            for i in range(len(correlations.columns)):
                for j in range(len(correlations.columns)):
                    axes[1, 1].text(j, i, f'{correlations.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
            
            plt.colorbar(im, ax=axes[1, 1])
            plt.tight_layout()
            plt.savefig('preference_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Statistical analysis
            print("\nStatistical Analysis:")
            print("-" * 40)
            
            # Use cleaned data for median calculation
            exp_median = corr_data['experience_years'].median()
            high_exp = corr_data[corr_data['experience_years'] > exp_median]
            low_exp = corr_data[corr_data['experience_years'] <= exp_median]
            
            print(f"High experience candidates (> {exp_median:.1f} years):")
            print(f"  • Average match score: {high_exp['matched_score_clean'].mean():.3f}")
            print(f"  • Standard deviation: {high_exp['matched_score_clean'].std():.3f}")
            print(f"  • Number of candidates: {len(high_exp)}")
            
            print(f"\nLow experience candidates (≤ {exp_median:.1f} years):")
            print(f"  • Average match score: {low_exp['matched_score_clean'].mean():.3f}")
            print(f"  • Standard deviation: {low_exp['matched_score_clean'].std():.3f}")
            print(f"  • Number of candidates: {len(low_exp)}")
            
            # Skill count analysis
            skill_median = corr_data['num_candidate_skills'].median()
            high_skills = corr_data[corr_data['num_candidate_skills'] > skill_median]
            low_skills = corr_data[corr_data['num_candidate_skills'] <= skill_median]
            
            print(f"\nHigh skill candidates (> {skill_median:.1f} skills):")
            print(f"  • Average match score: {high_skills['matched_score_clean'].mean():.3f}")
            
            print(f"\nLow skill candidates (≤ {skill_median:.1f} skills):")
            print(f"  • Average match score: {low_skills['matched_score_clean'].mean():.3f}")
            
        else:
            print("Insufficient data for correlation analysis.")
    else:
        print(f"Not enough valid match scores ({valid_scores}). Need at least 10 for meaningful analysis.")
else:
    print("\n'matched_score' column not found in the dataset.")
    print("Skipping preference analysis as match scores are not available.")

# ======================================================================
# ADDITIONAL INSIGHTS
# ======================================================================
print("\n" + "="*60)
print("ADDITIONAL INSIGHTS")
print("="*60)

# 1. Skills gap analysis
print("\n1. SKILLS GAP ANALYSIS")
print("-" * 40)

if 'skill_counter' in locals() and 'candidate_skill_counter' in locals():
    # Find skills that are in demand but not common among candidates
    demand_skills_set = set([skill for skill, _ in top_20_demand_skills]) if 'top_20_demand_skills' in locals() else set()
    common_skills_set = set([skill for skill, _ in top_20_common_skills]) if 'top_20_common_skills' in locals() else set()
    skills_gap = demand_skills_set - common_skills_set
    
    if skills_gap:
        print("Skills in high demand but less common among candidates:")
        for skill in sorted(skills_gap):
            demand_count = skill_counter.get(skill, 0)
            supply_count = candidate_skill_counter.get(skill, 0)
            print(f"  • {skill}:")
            print(f"    Demand: {demand_count} job postings")
            print(f"    Supply: {supply_count} candidates")
            print(f"    Gap: {demand_count - supply_count}")
    else:
        print("No significant skills gap found in top 20 skills.")
else:
    print("Skill counters not available for gap analysis.")

# 2. Role clustering by skills
print("\n2. ROLE CLUSTERING BY SKILLS")
print("-" * 40)

if role_skills:
    role_skill_counts = {}
    for role, skills in role_skills.items():
        unique_skills = set(skills)
        role_skill_counts[role] = len(unique_skills)
    
    print("\nTop roles by number of required skills:")
    sorted_roles = sorted(role_skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for role, count in sorted_roles:
        print(f"  • {role:40} - {count:3} unique skills required")
else:
    print("No role-skill data available for clustering analysis.")

# 3. Word cloud for skills
print("\n3. GENERATING WORD CLOUD FOR SKILLS")
print("-" * 40)

if all_required_skills:
    all_skills_text = ' '.join(all_required_skills)
    wordcloud = WordCloud(width=1200, height=600, 
                         background_color='white',
                         max_words=150,
                         contour_width=2,
                         contour_color='steelblue',
                         colormap='viridis').generate(all_skills_text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of In-Demand Skills', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('skills_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No skills data available for word cloud generation.")

# 4. Experience vs Skills Analysis
print("\n4. EXPERIENCE VS SKILLS ANALYSIS")
print("-" * 40)

# Group by experience levels
df['experience_group'] = pd.cut(df['experience_years'], 
                                bins=[0, 2, 5, 10, 20, 50],
                                labels=['0-2 yrs', '3-5 yrs', '6-10 yrs', '11-20 yrs', '20+ yrs'])

exp_skill_summary = df.groupby('experience_group').agg({
    'num_candidate_skills': ['mean', 'std', 'count'],
    'skill_match_ratio': ['mean', 'std']
}).round(2)

if not exp_skill_summary.empty:
    print("\nExperience Level vs Skills Summary:")
    print(exp_skill_summary)
else:
    print("Insufficient data for experience vs skills analysis.")

# ======================================================================
# SUMMARY REPORT
# ======================================================================
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

print("\nKEY FINDINGS:")
print("-" * 40)

# Top 3 most in-demand skills
if 'top_20_demand_skills' in locals() and top_20_demand_skills:
    print(f"1. Top 3 Most In-Demand Skills:")
    for i, (skill, count) in enumerate(top_20_demand_skills[:3], 1):
        percentage = (count / len(df)) * 100
        print(f"   {i}. {skill}: {count} jobs ({percentage:.1f}% of postings)")
else:
    print(f"1. Most In-Demand Skills: Data not available")

# Top 3 most common candidate skills
if 'top_20_common_skills' in locals() and top_20_common_skills:
    print(f"\n2. Top 3 Most Common Candidate Skills:")
    for i, (skill, count) in enumerate(top_20_common_skills[:3], 1):
        percentage = (count / len(df)) * 100
        print(f"   {i}. {skill}: {count} candidates ({percentage:.1f}% of candidates)")
else:
    print(f"\n2. Most Common Candidate Skills: Data not available")

# Average experience requirement
if 'experience_years' in df.columns and not df['experience_years'].isna().all():
    avg_exp = df['experience_years'].mean()
    print(f"\n3. Average Experience Requirement: {avg_exp:.1f} years")
    
    # Experience distribution
    exp_stats = df['experience_years'].describe()
    print(f"   • Min: {exp_stats['min']:.0f} years, Max: {exp_stats['max']:.0f} years")
    print(f"   • 25th percentile: {exp_stats['25%']:.1f} years")
    print(f"   • 75th percentile: {exp_stats['75%']:.1f} years")
else:
    print(f"\n3. Experience Requirements: Data not available")

# Most frequent education requirement
if 'all_education' in locals() and all_education:
    education_counter = Counter(all_education)
    if education_counter:
        most_common_edu = education_counter.most_common(1)[0]
        print(f"\n4. Most Common Education Requirement: {most_common_edu[0]} "
              f"({most_common_edu[1]} postings, {(most_common_edu[1]/len(df)*100):.1f}%)")
    else:
        print(f"\n4. Education Requirements: Data not available")
else:
    print(f"\n4. Education Requirements: Data not available")

# Skills gap analysis
if 'skills_gap' in locals() and skills_gap:
    print(f"\n5. Skills Gap Analysis: {len(skills_gap)} skills in high demand but low supply")
    if len(skills_gap) > 0:
        print("   Top skill gaps:")
        gap_details = []
        for skill in list(skills_gap)[:3]:
            demand = skill_counter.get(skill, 0)
            supply = candidate_skill_counter.get(skill, 0)
            gap_details.append(f"{skill} (D:{demand}, S:{supply})")
        print("   " + ", ".join(gap_details))
else:
    print(f"\n5. Skills Gap Analysis: Data not available")

# Correlation insights
if 'correlations' in locals():
    if 'skill_match_ratio' in correlations.index and 'matched_score_clean' in correlations.columns:
        skill_match_corr = correlations.loc['skill_match_ratio', 'matched_score_clean']
        print(f"\n6. Skill Match Ratio vs Match Score: {skill_match_corr:.3f}")
        if skill_match_corr > 0.3:
            print("   → Strong positive correlation: Better skill matches lead to higher scores")
        elif skill_match_corr > 0.1:
            print("   → Moderate positive correlation")
        else:
            print("   → Weak correlation")
    
    if 'experience_years' in correlations.index and 'matched_score_clean' in correlations.columns:
        exp_corr = correlations.loc['experience_years', 'matched_score_clean']
        print(f"   Experience vs Match Score: {exp_corr:.3f}")
        if exp_corr > 0.3:
            print("   → Strong positive correlation: Higher experience leads to higher scores")
        elif exp_corr > 0.1:
            print("   → Moderate positive correlation")
        else:
            print("   → Weak correlation")

# Data quality metrics
print(f"\n7. DATA QUALITY METRICS:")
print(f"   • Total records analyzed: {len(df):,}")
print(f"   • Records with candidate skills: {df['candidate_skills_clean'].apply(lambda x: len(x) > 0).sum():,}")
print(f"   • Records with required skills: {df['required_skills_clean'].apply(lambda x: len(x) > 0).sum():,}")
print(f"   • Records with experience data: {df['experience_years'].notna().sum():,}")
print(f"   • Unique job roles: {len(role_skills) if 'role_skills' in locals() else 0}")
print(f"   • Unique skills identified: {len(set(all_required_skills)) if 'all_required_skills' in locals() else 0}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)

# List generated files
print("\nGenerated visualizations:")
files_to_check = [
    'skills_by_role.png',
    'most_in_demand_skills.png',
    'most_common_skills_candidates.png',
    'education_requirements.png',
    'experience_distribution.png',
    'preference_analysis.png',
    'skills_wordcloud.png'
]

for file in files_to_check:
    import os
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} (not generated)")

print("\n" + "="*60)
print("RECOMMENDATIONS FOR USE:")
print("="*60)
print("""
1. Use the generated PNG files in reports or presentations
2. For Word document creation, copy key findings from SUMMARY REPORT
3. To automate Word document generation:
   - Install: pip install python-docx
   - Use the provided create_word_report() function
4. For further analysis:
   - Consider adding more data preprocessing steps
   - Explore machine learning for predictive analytics
   - Add industry-specific skill taxonomies
""")