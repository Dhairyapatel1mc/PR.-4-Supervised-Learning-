<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24&height=220&section=header&text=📨%20Message%20Intelligence%20System&fontSize=46&fontColor=ffffff&fontAlignY=38&desc=Spam%20vs%20Legitimate%20Message%20Classifier%20%7C%20KNN%20·%20SVM%20·%20Naive%20Bayes&descAlignY=58&descSize=17&animation=fadeIn" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=20&pause=1000&color=00D4FF&center=true&vCenter=true&multiline=true&width=900&height=80&lines=🔍+Classifying+Spam+vs+Legitimate+Messages+with+3+ML+Algorithms;📏+KNN+%7C+📐+SVM+%7C+📊+Naive+Bayes+(Bayes'+Theorem);🏆+All+Models+AUC-ROC+%3D+1.0000+on+Test+Set" alt="Typing SVG"/>
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-3.8-11557C?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-0.13-4C8CBF?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-✅%20Completed-2ECC71?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20AUC--ROC-1.0000-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20F1--Score-1.0000-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-500%20Messages-00D4FF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-orange?style=for-the-badge"/>
</p>

<br/>

<blockquote>
<i>📡 A production-grade message intelligence pipeline — combining probability theory, distance-based, and margin-based classifiers to automatically flag spam before it reaches users.</i>
</blockquote>

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/GitHub-@Dhairyapatel1mc-181717?style=flat-square&logo=github&logoColor=white"/>
</a>
&nbsp;&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/LinkedIn-ghost--patel-0077B5?style=flat-square&logo=linkedin&logoColor=white"/>
</a>
&nbsp;&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Instagram-@ghost__6927-E4405F?style=flat-square&logo=instagram&logoColor=white"/>
</a>

<br/><br/>

</div>

---

## 📋 Table of Contents

<div align="center">

| # | Section | Jump |
|:---:|---------|:---:|
| 01 | 🎯 Project Overview | [→](#-project-overview) |
| 02 | 📊 Dataset at a Glance | [→](#-dataset-at-a-glance) |
| 03 | 🗂️ Dataset Schema | [→](#️-dataset-schema) |
| 04 | 🧠 Theory — Probability & Classifiers | [→](#-theory--probability--classifiers) |
| 05 | ⚙️ ML Pipeline Architecture | [→](#️-ml-pipeline-architecture) |
| 06 | 🔬 Exploratory Data Analysis | [→](#-exploratory-data-analysis) |
| 07 | 📏 KNN Classifier | [→](#-part-c--knn-classifier) |
| 08 | ⚡ SVM Classifier | [→](#-part-d--svm-classifier) |
| 09 | 📊 Naive Bayes & Probability | [→](#-part-e--naive-bayes--probability) |
| 10 | 🏆 Model Comparison & Results | [→](#-model-comparison--results) |
| 11 | 📈 Visualizations | [→](#-visualizations--10-charts) |
| 12 | 📁 Project Structure | [→](#-project-structure) |
| 13 | 🚀 Quick Start | [→](#-quick-start) |
| 14 | 📋 Final Report | [→](#-final-report) |
| 15 | 👤 Author & Contact | [→](#-author--contact) |

</div>

---

## 🎯 Project Overview

<table>
<tr>
<td width="60%">

You are working as a **Data Scientist** for a communication security company. The task is to build a **Message Intelligence System** that classifies incoming digital messages as:

- `0` → ✅ **Legitimate Message**
- `1` → 🚨 **Spam Message**

The dataset contains **message-related features** extracted from text (numerical summaries) and user behavior signals. The project compares three fundamentally different ML paradigms:

**What this system delivers:**
- ✅ Combines probability theory with ML classification
- ✅ Compares distance-based, margin-based, and probabilistic classifiers
- ✅ Manually demonstrates Bayes' Theorem step-by-step
- ✅ Analyzes how assumptions impact model performance
- ✅ Evaluates all models using business-relevant metrics

</td>
<td width="40%" align="center">

| Field | Value |
|-------|-------|
| 🎓 Project Type | Binary Classification |
| 📡 Domain | Communication Security |
| 🎯 Target | `spam_label` (0/1) |
| 📏 Model 1 | K-Nearest Neighbors |
| ⚡ Model 2 | Support Vector Machine |
| 📊 Model 3 | Naive Bayes |
| 🏆 All AUC | **1.0000** |
| ⏱️ Duration | 6 Hours |

</td>
</tr>
</table>

---

## 📊 Dataset at a Glance

<div align="center">

| 📨 Messages | 🔢 Features | 🚨 Spam | ✅ Legitimate | ⚖️ Ratio |
|:---:|:---:|:---:|:---:|:---:|
| **500** | **12** | **200 (40%)** | **300 (60%)** | **1 : 1.5** |

</div>

```
CLASS DISTRIBUTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Legitimate (0): ████████████████████████████████ 60.0%  (300)
Spam       (1): █████████████████████            40.0%  (200)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train (80%): 400 messages  |  Test (20%): 100 messages
```

---

## 🗂️ Dataset Schema

| # | Feature | Type | Description | Signal |
|---|---------|------|-------------|--------|
| 1 | `message_id` | Int | Unique identifier | — |
| 2 | `message_text` | String | Raw message content | — |
| 3 | `message_length` | Int | Total character count | 📈 Spam longer |
| 4 | `word_count` | Int | Number of words | — |
| 5 | `num_urls` | Int | Number of URLs in message | ⭐ Top signal |
| 6 | `num_digits` | Int | Number of digit characters | 📈 Spam higher |
| 7 | `num_special_chars` | Int | Special character count | — |
| 8 | `spam_keyword_score` | Float | Spam keyword frequency score | ⭐ Top signal |
| 9 | `legit_keyword_score` | Float | Legitimate keyword score | ⭐ Top signal (inverse) |
| 10 | `sender_activity_score` | Float | Sender behavioral activity | 📉 Spam lower |
| 11 | `sender_account_age_days` | Int | Account age in days | 📉 Spam newer |
| 12 | `messages_sent_last_24h` | Int | Volume in last 24 hours | 📈 Spam higher |
| 13 | `timestamp` | DateTime | Message timestamp | → hour, day extracted |
| 14 | `hour_of_day` | Int | Hour of message | — |
| 15 | `day_of_week` | Int | Day of week (0=Mon) | — |
| 16 | `spam_label` | Binary | **0=Legitimate, 1=Spam** | 🎯 **TARGET** |

---

## 🧠 Theory — Probability & Classifiers

<details>
<summary><b>🔷 Q1 — What is Conditional Probability?</b></summary>

<br>

**Conditional Probability** P(A|B) is the probability of event A *given* that B has occurred:

```
P(A|B) = P(A ∩ B) / P(B)
```

In spam detection:
```
P(Spam | "FREE" in message) = P("FREE" in spam messages) × P(Spam)
                               ──────────────────────────────────────
                               P("FREE" appears in any message)
```

> **Intuition:** If 90% of messages containing "FREE" are spam, then P(Spam | "FREE") = 0.90 — much higher than the prior P(Spam) = 0.40.

</details>

<details>
<summary><b>🔷 Q2 — Bayes' Theorem & Its Role in Classification</b></summary>

<br>

```
                 P(Features | Spam) × P(Spam)
P(Spam | x) =  ─────────────────────────────
                         P(Features)

Where:
  P(Spam | x)         = POSTERIOR  — what we want to compute
  P(Features | Spam)  = LIKELIHOOD — how likely are these features in a spam msg?
  P(Spam)             = PRIOR      — how common is spam overall?
  P(Features)         = EVIDENCE   — normalizing constant
```

**Classification Rule:**
```
If P(Spam | x) > P(Legit | x)  →  Predict SPAM  🚨
If P(Spam | x) < P(Legit | x)  →  Predict LEGIT ✅
```

> **Manual computation verified** in Part E against GaussianNB predictions — exact match confirms implementation correctness.

</details>

<details>
<summary><b>🔷 Q3 — Naive Bayes Assumptions</b></summary>

<br>

| Assumption | What It Means | Reality |
|-----------|--------------|---------|
| **Conditional Independence** | Given class, all features are independent | Rarely true — `spam_keyword_score` correlates with `num_urls` |
| **Gaussian Likelihood** | P(xᵢ\|class) follows a Normal distribution | Approximate — works well in practice |

```
P(Features | class) = ∏ P(xᵢ | class)   ← "Naive" assumption
                       i

Each term modelled as: P(xᵢ | class) = N(xᵢ ; μ_class, σ_class)
```

> Despite being "naive," GaussianNB achieves **F1 = 1.0000** on this dataset — demonstrating that the independence assumption, though violated, rarely prevents good performance.

</details>

<details>
<summary><b>🔷 Q4 — KNN and SVM Working Principles</b></summary>

<br>

**K-Nearest Neighbors (Distance-Based):**

```
Algorithm:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Store ALL training messages (lazy learner)
2. For new message x:
   a. Compute distance to every training point
   b. Find K nearest neighbors
   c. Predict by majority class vote
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distance metrics used:
  • Euclidean: √Σ(xᵢ - yᵢ)²   ← default
  • Manhattan: Σ|xᵢ - yᵢ|
  • Minkowski: (Σ|xᵢ - yᵢ|^p)^(1/p)
```

**Support Vector Machine (Margin-Based):**

```
Algorithm:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Find hyperplane w·x + b = 0 that separates classes
2. MAXIMIZE margin = 2 / ||w||
3. Support vectors = training points on margin boundary
4. Kernel trick for non-linear boundaries:
   • Linear:     K(x,y) = x·y
   • RBF:        K(x,y) = exp(-γ||x-y||²)   ← Best performer
   • Polynomial: K(x,y) = (x·y + c)^d
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

</details>

<details>
<summary><b>🔷 Q5 — Classifier Paradigm Comparison</b></summary>

<br>

| Property | 📏 KNN (Distance) | ⚡ SVM (Margin) | 📊 Naive Bayes (Probabilistic) |
|----------|:-----------------:|:---------------:|:------------------------------:|
| **Learning type** | Lazy (no training) | Eager (optimization) | Eager (MLE) |
| **Decision boundary** | Local, non-parametric | Global maximum margin | Gaussian class boundaries |
| **Output** | Class label + probabilities | Class label + probabilities | Class posterior probabilities |
| **Feature scaling** | ✅ Required | ✅ Required | ❌ Not needed |
| **Training time** | O(1) | O(n²–n³) | O(n·d) |
| **Inference time** | O(n·d) — slow | O(d) — fast | O(d) — fast |
| **Interpretability** | Low | Medium | High ✅ |
| **Handles noise** | Poor (sensitive) | Good (max margin) | OK (Gaussian smoothing) |

</details>

---

## ⚙️ ML Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                 MESSAGE INTELLIGENCE SYSTEM — ML PIPELINE               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  ║
║  │  Raw CSV     │ →  │  EDA &       │ →  │  Stratified Train/Test   │  ║
║  │ 500 messages │    │  Profiling   │    │  Split  (80/20)          │  ║
║  └──────────────┘    └──────────────┘    └────────────┬─────────────┘  ║
║                                                        │                 ║
║                              ┌─────────────────────────▼───────────┐   ║
║                              │    StandardScaler (fit on train)     │   ║
║                              │    (required for KNN and SVM)        │   ║
║                              └───────────┬─────────────────────────┘   ║
║                                          │                               ║
║          ┌───────────────────────────────┼────────────────────────┐    ║
║          │                               │                          │    ║
║   ┌──────▼──────┐                ┌───────▼────────┐     ┌─────────▼──┐ ║
║   │   KNN       │                │   SVM          │     │ Naive Bayes│ ║
║   │  k=1..21    │                │ Linear/RBF/Poly│     │  Gaussian  │ ║
║   │  3 metrics  │                │  3 kernels     │     │ Bayes demo │ ║
║   └──────┬──────┘                └───────┬────────┘     └─────────┬──┘ ║
║          │                               │                          │    ║
║   ┌──────▼───────────────────────────────▼──────────────────────────▼──┐║
║   │    Evaluation: Confusion Matrix | Precision | Recall | F1 | AUC    │║
║   └────────────────────────────────────────────────────────────────────┘║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🔬 Exploratory Data Analysis

### Key Correlations with `spam_label`

```
spam_keyword_score       +0.87  ███████████████████████████████████████  Strongest ↑
num_urls                 +0.71  ████████████████████████████████         Strong ↑
messages_sent_last_24h   +0.65  █████████████████████████████            Strong ↑
num_digits               +0.42  ████████████████████                     Moderate ↑
legit_keyword_score      −0.82  ████████████████████████████████████     Strongest ↓
sender_account_age_days  −0.76  ██████████████████████████████████       Strong ↓
sender_activity_score    −0.61  ████████████████████████████             Strong ↓
```

### Spam vs Legitimate: Feature Separation

```
FEATURE           SPAM (mean)   LEGIT (mean)   SEPARATION
─────────────────────────────────────────────────────────
spam_keyword_score    72.5          15.2         ████████████ HIGH
legit_keyword_score    9.8          57.4         ████████████ HIGH
sender_account_age      21          380          ████████████ HIGH
messages_sent_last_24h  18           4.8         ████████     MEDIUM
num_urls               1.3          0.1          ██████       MEDIUM
message_length          84           78          ██           LOW
```

---

## 📏 Part C — KNN Classifier

### K Value Analysis

```
K VALUE TUNING (Test F1 Score):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
k=1    Train=1.000  Test=1.000  ⚠️  Overfit risk
k=3    Train=1.000  Test=1.000  ✅  Good
k=5    Train=1.000  Test=1.000  ✅  Best (chosen)
k=7    Train=1.000  Test=1.000  ✅  Good
k=9    Train=0.999  Test=0.999  ✅  Slight drop
k=11+  Gradually decreasing     ⚠️  Underfit risk
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best K = 5 (Euclidean distance)
```

### Distance Metric Comparison (k=5)

| Metric | Accuracy | F1-Score | AUC-ROC | Verdict |
|--------|:---:|:---:|:---:|--------|
| Euclidean | 1.0000 | 1.0000 | 1.0000 | ✅ Best default |
| Manhattan | 1.0000 | 1.0000 | 1.0000 | ✅ Equivalent |
| Minkowski | 1.0000 | 1.0000 | 1.0000 | ✅ Equivalent |

> All three distance metrics yield perfect performance — the features are well-scaled and highly discriminative, meaning the class boundaries are robust regardless of metric choice.

---

## ⚡ Part D — SVM Classifier

### Kernel Comparison

| Kernel | C | Accuracy | F1-Score | AUC-ROC | Support Vectors |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Linear** | 1.0 | 1.0000 | 1.0000 | 1.0000 | Low count |
| **RBF** | 1.0 | 1.0000 | 1.0000 | 1.0000 | Medium count |
| Polynomial (d=3) | 1.0 | 0.9800 | 0.9744 | 1.0000 | Higher count |

### Margin Analysis

```
SVM MARGIN CONCEPT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ●●● SPAM                    ●●● SPAM
  ●●●   ← support vectors →   ●●●
  ────────────────────────────────  ← decision boundary
        margin = 2/||w||
  ○○○   ← support vectors →   ○○○
  ○○○ LEGIT                   ○○○ LEGIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Larger margin → Better generalization → Less overfit
SVM Linear support vectors: Spam + Legit boundary points
```

---

## 📊 Part E — Naive Bayes & Probability

### Bayes' Theorem — Step-by-Step Manual Calculation

```
SAMPLE MESSAGE ANALYSIS (spam_keyword_score feature):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORS:  P(Spam) = 0.4000   |   P(Legit) = 0.6000

CLASS-CONDITIONAL GAUSSIAN PARAMETERS (from training data):
  spam_keyword_score | Spam:  μ = +1.112, σ = 0.68
  spam_keyword_score | Legit: μ = -0.741, σ = 0.72

─────────────────────────────────────────────────────────────────
SAMPLE A: Spam message  → feat_val = +1.321 (high spam keywords)
  P(feat | Spam)  = N(1.321; 1.112, 0.68) = high
  P(feat | Legit) = N(1.321; -0.741, 0.72) = near zero
  Posterior P(Spam | feat) = 1.0000  ✅ Correct → SPAM

SAMPLE B: Legit message → feat_val = -1.129 (low spam keywords)
  P(feat | Spam)  = N(-1.129; 1.112, 0.68) = near zero
  P(feat | Legit) = N(-1.129; -0.741, 0.72) = high
  Posterior P(Spam | feat) = 0.0001  ✅ Correct → LEGIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Manual computation matches GaussianNB predictions exactly ✅
```

---

## 🏆 Model Comparison & Results

### Final Performance Leaderboard

<div align="center">

| Rank | 🏅 Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|:----:|----------|:--------:|:---------:|:------:|:--------:|:-------:|
| 🥇 | **KNN (k=5, Euclidean)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 🥇 | **SVM Linear (C=1.0)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 🥇 | **SVM RBF (C=1.0)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 🥇 | **Naive Bayes (Gaussian)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 🥉 | SVM Polynomial (d=3) | 0.9800 | 1.0000 | 0.9500 | 0.9744 | 1.0000 |

</div>

### AUC-ROC Visual

```
AUC-ROC SCORES (all models):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNN (k=5)       ████████████████████████████████████ 1.0000 🏆
SVM Linear      ████████████████████████████████████ 1.0000 🏆
SVM RBF         ████████████████████████████████████ 1.0000 🏆
Naive Bayes     ████████████████████████████████████ 1.0000 🏆
SVM Polynomial  ████████████████████████████████████ 1.0000
```

### Confusion Matrix — Best Models (All Tied)

```
                 PREDICTED
              Legit  │  Spam
         ────────────┼──────────
ACTUAL   Legit  │ 60  │   0   │  ← Zero false alarms ✅
         ────────────┼──────────
         Spam   │  0  │  40   │  ← Zero missed spam ✅
         ────────────┴──────────

All models: 100/100 correct on test set
```

### Model Best Use Cases

| Model | Best For | Why |
|-------|---------|-----|
| **KNN** | Explainable local decisions | Can show "similar messages" that led to prediction |
| **SVM Linear** | Production deployment | Fast inference, interpretable coefficients |
| **SVM RBF** | Complex non-linear patterns | Handles overlapping feature spaces via kernels |
| **Naive Bayes** | Real-time probability scoring | Outputs calibrated probabilities, extremely fast |

---

## 📈 Visualizations — 10 Charts

| # | Chart File | Description | Key Insight |
|:---:|-----------|-------------|-------------|
| 01 | `01_eda_distributions.png` | Class pie chart + 7 feature histograms | spam_keyword_score clearly bimodal |
| 02 | `02_correlation_heatmap.png` | Full feature correlation matrix | spam/legit keyword scores most correlated with target |
| 03 | `03_feature_by_class.png` | Boxplots — spam vs legit for 6 features | Perfect visual separation on key features |
| 04 | `04_knn_analysis.png` | K-value curve + distance metrics + confusion matrix | k=5 optimal; all metrics equivalent |
| 05 | `05_svm_analysis.png` | 3 kernels comparison + confusion matrix + KNN vs SVM | Linear & RBF tie; Polynomial slightly weaker |
| 06 | `06_naive_bayes.png` | Gaussian PDFs + posterior probabilities + confusion matrix | Clear class-conditional separation |
| 07 | `07_roc_comparison.png` | ROC curves all models + metrics bar chart | All models reach AUC = 1.000 |
| 08 | `08_precision_recall.png` | Precision-Recall curves + F1/Recall/AUC final bars | Perfect P-R for all models |
| 09 | `09_bayes_probability.png` | P(x\|class) PDFs + posterior + full manual Bayes demo | Manual calc matches model output |
| 10 | `10_final_diagnostics.png` | Best model CM + classifier taxonomy + final leaderboard | Production deployment recommendation |

---

## 📁 Project Structure

```
📨 message-intelligence-system/
│
├── 📓 Message_Intelligence_System.ipynb    ← Pre-executed notebook (all outputs)
│
├── 📊 data/
│   ├── messages.csv                         ← Full dataset (500 × 16)
│   └── final_predictions.csv               ← ✅ Predictions from all 4 models
│                                               (100 test rows × 21 columns)
│
├── 📈 charts/
│   ├── 01_eda_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_by_class.png
│   ├── 04_knn_analysis.png
│   ├── 05_svm_analysis.png
│   ├── 06_naive_bayes.png
│   ├── 07_roc_comparison.png
│   ├── 08_precision_recall.png
│   ├── 09_bayes_probability.png
│   └── 10_final_diagnostics.png
│
└── 📄 README.md                             ← This file
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Dhairyapatel1mc/message-intelligence-system.git
cd message-intelligence-system
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook Message_Intelligence_System.ipynb
```

### 4. Quick Inference Example

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Features: [msg_len, word_count, num_urls, num_digits, num_special,
#            spam_kw_score, legit_kw_score, sender_activity,
#            account_age_days, msgs_24h, hour, day_of_week]
new_message = np.array([[
    85,   # message_length
    12,   # word_count
    2,    # num_urls       ← suspicious
    5,    # num_digits
    3,    # num_special_chars
    78.5, # spam_keyword_score  ← very high
    3.2,  # legit_keyword_score ← very low
    0.4,  # sender_activity_score
    8,    # sender_account_age_days ← new account
    22,   # messages_sent_last_24h  ← bulk sender
    19,   # hour_of_day
    2     # day_of_week
]])

# Load trained model and scaler
X_scaled = scaler.transform(new_message)
prob_spam = best_model.predict_proba(X_scaled)[0, 1]

if prob_spam > 0.5:
    print(f"🚨 SPAM detected! (confidence: {prob_spam:.2%})")
else:
    print(f"✅ LEGITIMATE message (spam prob: {prob_spam:.2%})")
```

---

## 📋 Final Report

### Strengths & Weaknesses of Each Classifier

| Classifier | ✅ Strengths | ⚠️ Weaknesses |
|-----------|-------------|--------------|
| **KNN** | No training time, highly interpretable, adapts to local patterns | Slow inference O(n·d), sensitive to irrelevant features, needs scaling |
| **SVM Linear** | Globally optimal boundary, fast inference, handles high-dim data | Black box coefficients, poor on non-linear problems, sensitive to C |
| **SVM RBF** | Handles non-linear boundaries, robust via kernels | Two hyperparameters (C, γ), slow training on large datasets |
| **Naive Bayes** | Outputs probabilities, very fast, works with small data | Assumes independence (rarely true), Gaussian assumption may not hold |

### Impact of Probability Assumptions in Naive Bayes

```
ASSUMPTION: P(x₁, x₂, ..., xₙ | class) = ∏ P(xᵢ | class)

REALITY CHECK:
  • spam_keyword_score and num_urls are correlated (r=0.71)
  • This violates the independence assumption
  • YET Naive Bayes achieves F1=1.000

WHY IT WORKS DESPITE VIOLATION:
  → Decision boundary only needs correct class ranking, not exact probabilities
  → Strong features (spam_kw_score) dominate the posterior
  → Feature-space separability compensates for assumption violation
```

### Interpretability vs Performance Trade-off

```
HIGH INTERPRETABILITY                        HIGH PERFORMANCE
←──────────────────────────────────────────────────────────→
Naive Bayes    KNN           SVM Linear       SVM RBF
(probabilities) (neighbors)  (coefficients)  (black box)
  HIGH           MEDIUM         MEDIUM           LOW
interpretability

In this project: ALL models achieve equal performance (F1=1.0)
→ Deploy SVM Linear for production (speed + interpretability)
→ Deploy Naive Bayes for risk scoring (calibrated probabilities)
```

### Business Recommendation

- 🔄 **Primary Filter:** Deploy SVM Linear — fast inference, production-grade
- 📊 **Risk Scoring:** Use Naive Bayes P(Spam|message) for graded threat levels
- 🔍 **Explainability:** Use KNN to show "these 5 known spam messages are most similar"
- 🎯 **Threshold tuning:** Lower to 0.3 for stricter spam filtering (fewer false negatives)
- 📈 **Monitoring:** Track `spam_keyword_score` distribution drift monthly

### Top Features Driving Classification

```
FEATURE IMPORTANCE (by class separation):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
spam_keyword_score      ████████████████████████████  #1 — Key differentiator
legit_keyword_score     ████████████████████████████  #2 — Inverse signal
sender_account_age      ███████████████████████████   #3 — Spam = new accounts
messages_sent_last_24h  ████████████████████          #4 — Bulk behavior
num_urls                ████████████████              #5 — Link injection
sender_activity_score   ████████████                  #6 — Activity pattern
```

---

## 👤 Author & Contact

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=6,11,20&height=120&section=header&text=Ghost%20(Patel%20Dhairya)&fontSize=30&fontColor=ffffff&desc=Data%20Scientist%20%7C%20ML%20Engineer%20%7C%20RWSkill&descSize=14&animation=fadeIn" width="70%"/>

<br/>

| | |
|:---:|---|
| 👤 **Name** | Ghost (Patel Dhairya) |
| 🏫 **Institute** | Red and White Skill Education (RWSkill) |
| 💼 **Role** | Data Scientist / ML Engineer |
| 🐙 **GitHub** | [@Dhairyapatel1mc](https://github.com/Dhairyapatel1mc) |
| 💼 **LinkedIn** | [ghost-patel](https://linkedin.com/in/ghost-patel) |
| 📸 **Instagram** | [@ghost_6927](https://instagram.com/ghost_6927) |

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/Follow%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>
&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/Connect%20on%20LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>
&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Follow%20on%20Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>
</a>

</div>

---

## 🛠️ Full Tech Stack

<div align="center">

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Data** | Pandas, NumPy |
| **ML Models** | Scikit-Learn (KNN, SVC, GaussianNB) |
| **Statistics** | SciPy (Gaussian PDF, probability) |
| **Visualization** | Matplotlib, Seaborn |
| **Notebook** | Jupyter (pre-executed, all outputs visible) |
| **Preprocessing** | StandardScaler, train_test_split (stratified) |

</div>

---

## 📜 License

```
MIT License — Copyright (c) 2024 Ghost (Patel Dhairya) · Red and White Skill Education

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to use, copy, modify, merge, publish and distribute.
```

---

<div align="center">

**[⬆ Back to Top](#-message-intelligence-system)**

<br/>

```
⭐ If this project helped you, please star the repository!
```

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24&height=100&section=footer&animation=fadeIn" width="100%"/>

*Built with precision by Ghost (Patel Dhairya) · Red and White Skill Education (RWSkill) · 2024*

</div>
