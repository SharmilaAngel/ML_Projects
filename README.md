# 🧠 Customer Segmentation using Unsupervised Machine Learning

## 📌 Problem Statement

In a competitive business environment, understanding customer behavior is crucial for designing effective marketing strategies. However, businesses often struggle to treat customers as unique individuals due to large and diverse customer bases.  
This project aims to solve this by **segmenting customers** into distinct groups using **unsupervised machine learning**, enabling targeted campaigns and personalized services.

---

## 🛠️ Technologies and Tools Used

- **Python 3**
- **Pandas** – Data manipulation
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning (StandardScaler, KMeans, t-SNE, LabelEncoder)
- **Jupyter Notebook / Google Colab** – Development environment

---

## 🔁 Project Workflow

### 1. 📚 Importing the Libraries

All necessary Python libraries are imported including `pandas`, `numpy`, `seaborn`, `matplotlib`, and `scikit-learn`.

### 2. 📂 Load the Dataset

The dataset used (`new.csv`) contains customer demographic and transactional information. It is loaded and inspected for structure and completeness.

### 3. 🧹 Data Preprocessing

- Handling **missing values** (e.g., dropping rows with null `Income`)
- Extracting date features from `Dt_Customer` (day, month, year)
- Dropping irrelevant columns (`Z_CostContact`, `Z_Revenue`)
- **Label encoding** for categorical variables
- **Feature scaling** using `StandardScaler`

### 4. 📊 Data Visualization and Analysis

- Count plots for categorical features (e.g., `Education`, `Marital_Status`)
- Correlation heatmap to detect multicollinearity
- t-SNE used for dimensionality reduction and visualization

### 5. 🔎 Segmentation

- Applied **K-Means Clustering** on scaled data
- Used the **Elbow Method** to determine the optimal number of clusters
- Evaluated results using **inertia** and **silhouette score**
- Final segmentation performed with `k = 6` clusters

### 6. 🧬 Cluster Profiling

Each cluster is analyzed to understand the demographic and behavioral traits of the customers within it. This includes:
- Spending patterns
- Income groups
- Marital status
- Product preferences

### 7. 💡 Insights & Business Recommendations

- Identify **high-value customers** for loyalty programs
- Re-engage **infrequent buyers** through targeted promotions
- **Upsell** and **cross-sell** strategies based on purchasing behaviors
- Develop **personalized marketing** based on cluster attributes

---

## 📈 Future Work

- Use **DBSCAN** or **Hierarchical Clustering** for more flexible grouping
- Build a **real-time dashboard** to monitor cluster shifts
- Integrate data from other channels (web, mobile, feedback)
- Apply **predictive models** for churn and customer lifetime value (CLV)

---
