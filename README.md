# 📊 Resume Data Mining & Job Market Analysis System

A comprehensive data mining and analytics project that extracts meaningful insights from resume and job description data to understand skill demand, hiring trends, and candidate-job alignment.

---

## 🚀 Overview

This project performs end-to-end data mining on resume datasets to uncover patterns in hiring requirements, candidate skills, and job market trends.

It applies data preprocessing, feature engineering, statistical analysis, and visualization techniques to answer key business questions such as:

* Which skills are most in demand?
* What skills do candidates commonly have?
* What is the gap between demand and supply?
* How do experience and skills affect hiring preference?

---

## 🛠️ Tech Stack

* Python
* Pandas & NumPy (Data Processing)
* Matplotlib & Seaborn (Visualization)
* WordCloud (Text Mining)
* Regex & AST (Data Cleaning & Parsing)

---

## 📂 Dataset

The dataset includes:

* Candidate skills
* Required job skills
* Job roles
* Experience requirements
* Educational qualifications
* Matching scores (if available)

---

## 🔍 Key Features

* ✅ Data cleaning & preprocessing pipeline
* ✅ Skill extraction from unstructured text
* ✅ Feature engineering (skill count, match ratio)
* ✅ Demand vs supply gap analysis
* ✅ Role-based skill analysis
* ✅ Experience & education insights
* ✅ Correlation and regression analysis
* ✅ Automated visualization generation

---

## 📊 Data Mining Concepts Used

* **Association Analysis** → Role vs Required Skills
* **Frequency Analysis** → Most in-demand & common skills
* **Text Mining** → WordCloud visualization
* **Descriptive Statistics** → Experience & education trends
* **Correlation Analysis** → Hiring preferences
* **Feature Engineering** → Skill match ratio (Jaccard similarity)

---

## ⚙️ How It Works

### 1. Data Preprocessing

* Parses stringified lists (skills)
* Cleans noisy text data
* Extracts numerical experience values
* Handles missing values

---

### 2. Feature Engineering

* Number of candidate skills
* Skill match ratio (candidate vs job requirements)
* Cleaned education and role fields

---

### 3. Analysis Performed

### 📌 Skill vs Role Analysis

* Identifies top skills required for each job role
* Visualized using bar charts

---

### 📌 Demand vs Supply Analysis

* Most in-demand skills (job requirements)
* Most common candidate skills
* Gap between demand and supply

---

### 📌 Education & Experience Analysis

* Most required education levels
* Experience distribution across roles
* Outlier detection using boxplots

---

### 📌 Preference Analysis

* Correlation between:

  * Experience
  * Skills
  * Match score
* Regression and heatmap visualizations

---

## 📈 Output Visualizations

The system automatically generates:

* 📊 Skills by Role (`skills_by_role.png`)
* 📊 In-Demand Skills (`most_in_demand_skills.png`)
* 📊 Demand vs Supply Gap (`most_common_skills_candidates.png`)
* ☁️ Skills WordCloud (`skills_wordcloud.png`)
* 🎓 Education Requirements (`education_requirements.png`)
* 📉 Experience Distribution (`experience_distribution.png`)
* 🔥 Correlation Heatmap (`preference_analysis.png`)

---

## ▶️ Installation & Usage

1. Clone the repository:

```
git clone https://github.com/your-username/resume-data-mining.git
cd resume-data-mining
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Add your dataset:

```
resume_data.csv
```

4. Run the project:

```
python main.py
```

---

## 📌 Sample Insights (Example)

* Python, SQL, and Machine Learning are highly demanded skills
* Skill gaps exist between job requirements and candidate profiles
* Higher skill match ratio correlates with better hiring scores
* Experience shows moderate influence on hiring decisions

---

## ⚠️ Limitations

* Dependent on dataset quality
* Assumes consistent data formatting
* Limited NLP (no deep semantic understanding)

---

## 🔮 Future Improvements

* Integrate NLP models (BERT, embeddings)
* Build recommendation system (job ↔ candidate matching)
* Deploy as a web dashboard (Flask/Streamlit)
* Add real-time analytics
* Use larger real-world datasets

---

## 💡 Use Cases

* Recruitment analytics
* Resume screening systems
* HR decision support
* Skill gap analysis platforms
* Job market research

---

## 🧑‍💻 Author

Aryan Khalique

---

## ⭐ Contributing

Contributions are welcome! Feel free to fork the repository and improve it.

---

## 📜 License

This project is open-source and available under the MIT License.


<img width="975" height="547" alt="image" src="https://github.com/user-attachments/assets/a27b3a03-d5ce-45e0-8322-71846cb3f8ca" />


<img width="664" height="370" alt="image python 2" src="https://github.com/user-attachments/assets/3a9e6865-d903-46f0-9f74-9fd9ae056752" />


