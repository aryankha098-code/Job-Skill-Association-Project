import pandas as pd
import ast # For safely evaluating string representations of lists
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- Part 1: Data Loading, Cleaning, and Normalization ---

# Load Data
data = pd.read_csv("resume_data.csv")
data.index.name = 'resume_id'

# Normalization Function
def normalize(col):
    """Normalize text columns: lowercase, replace delimiters with comma, strip whitespace."""
    if col.dtype == 'object':
        return (col.str.lower()
                    .str.replace("|", ",", regex=False)
                    .str.replace(";", ",", regex=False)
                    .str.strip()
                    .replace('', pd.NA) # Replace empty strings with NA
        )
    return col

# Apply normalization
data["Skills"] = normalize(data["Skills"])
data["job_position_name"] = normalize(data["job_position_name"])

# Handle NaNs and split into lists
data['Skills'] = data['Skills'].fillna('unknown')
data['job_position_name'] = data['job_position_name'].fillna('unknown')

data["skills_list"] = data["Skills"].str.split(",")
data["job_list"] = data["job_position_name"].str.split(",")

print("--- Part 1 Complete: Initial data cleaning and feature splitting ---")


# --- Part 2: Skill-Job Association Analysis ---

# Explode and clean data for association analysis
df_skills = data.explode("skills_list").copy()
df_skills["skills_list"] = df_skills["skills_list"].str.strip()

df_skill_job = df_skills.explode("job_list").copy()
df_skill_job["job_list"] = df_skill_job["job_list"].str.strip()

# Rename columns and filter out noise
df_skill_job.rename(columns={
    "skills_list": "skill",
    "job_list": "job_role"
}, inplace=True)

df_skill_job = df_skill_job[
    (df_skill_job["skill"] != "") &
    (df_skill_job["job_role"] != "") &
    (df_skill_job["skill"] != "unknown")
]

# Calculate the association matrix (Skill vs Job Role)
skill_job_matrix = pd.crosstab(
    df_skill_job["skill"],
    df_skill_job["job_role"]
)

# Calculate the percentage of skill usage WITHIN each job role
skill_job_percentage = skill_job_matrix.div(
    skill_job_matrix.sum(axis=0),
    axis=1
) * 100

# Find the Top N skills per role (N=5)
N = 5
top_skills_per_role = {}
for job_role in skill_job_percentage.columns:
    top_skills = (
        skill_job_percentage[job_role]
        .sort_values(ascending=False)
        .head(N)
        .index.tolist()
    )
    top_skills_per_role[job_role] = top_skills

top_skills_df = pd.DataFrame(
    list(top_skills_per_role.items()),
    columns=["Job Role", f"Top {N} Skills (by Percentage)"]
)
top_skills_df.to_csv("top_skills_per_role.csv", index=False)

print("--- Part 2 Complete: Skill-Job Association calculated and saved to top_skills_per_role.csv ---")


# --- Part 3: Experience Analysis (Actual Experience Calculation) - FIXED CODE ---

date_cols = ['start_dates', 'end_dates']

# FIX: Robust function to handle various data types before explosion
def safe_convert_to_list(x):
    """
    Safely converts a cell value into a list format for explosion.
    Handles existing lists, NaN, string representations of lists, and single scalar values.
    """
    if isinstance(x, (list, tuple)):
        return list(x)

    if pd.isna(x):
        return []

    if isinstance(x, str) and x.strip().startswith('['):
        try:
            return ast.literal_eval(x)
        except:
            return [x]

    return [x]

# Apply the fixed conversion function
for col in date_cols:
    data[col] = data[col].apply(safe_convert_to_list)

# Explode the DataFrame for individual job entries
experience_df = data.explode(date_cols)
experience_df = experience_df.dropna(subset=['start_dates', 'end_dates'])

# Function to parse date strings (handling 'Till Date'/'Present')
def parse_experience_date(date_str):
    if isinstance(date_str, str):
        date_str_lower = date_str.lower().strip()
        if 'till date' in date_str_lower or 'present' in date_str_lower:
            return datetime.now()
        try:
            # Attempt to parse 'Month Year' format (e.g., Nov 2019)
            return datetime.strptime(date_str_lower, '%b %Y')
        except:
            return None
    return None

# Convert date strings to datetime objects and calculate duration
experience_df['start_dt'] = experience_df['start_dates'].apply(parse_experience_date)
experience_df['end_dt'] = experience_df['end_dates'].apply(parse_experience_date)

experience_df = experience_df.dropna(subset=['start_dt', 'end_dt'])

experience_df['duration_days'] = (experience_df['end_dt'] - experience_df['start_dt']).dt.days
experience_df = experience_df[experience_df['duration_days'] > 0] # Filter out negative/zero duration

# Sum total experience in years for each resume
total_experience = (
    experience_df.groupby('resume_id')['duration_days'].sum() / 365.25 # Convert days to years
).reset_index(name='total_experience_years')

total_experience.to_csv("total_experience_analysis_fixed.csv", index=False)

print("--- Part 3 Complete: Total experience calculated and saved to total_experience_analysis_fixed.csv ---")


# --- Part 4: Visualization (Heatmap) ---

# Select a subset for visualization: Top 10 most frequent jobs and Top 15 most frequent skills
# This step relies on 'skill_job_matrix' and 'skill_job_percentage' from Part 2.

try:
    # Top 10 jobs by frequency
    top_jobs = skill_job_matrix.sum(axis=0).nlargest(10).index
    skill_job_percentage_subset = skill_job_percentage[top_jobs]

    # Top 15 skills by frequency
    top_skills = skill_job_matrix.sum(axis=1).nlargest(15).index
    skill_job_percentage_subset = skill_job_percentage_subset.loc[top_skills]

    # Generate Heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        skill_job_percentage_subset,
        cmap="viridis",
        linewidths=0.5,
        annot=True,
        fmt=".1f" # Format annotations to 1 decimal place
    )
    plt.title("Top Skill–Job Role Association (Percentage)", fontsize=16)
    plt.ylabel("Skill", fontsize=14)
    plt.xlabel("Job Role", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("skill_job_heatmap.png")

    print("--- Part 4 Complete: Heatmap generated as skill_job_heatmap.png ---")

except NameError:
    print("--- Part 4 Note: Run Parts 1 and 2 first to define 'skill_job_matrix' and 'skill_job_percentage' for visualization. ---")