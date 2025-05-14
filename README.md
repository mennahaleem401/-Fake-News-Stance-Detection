# Fake-News-Stance-Detection
# **About the Dataset**
---

## 📘 Introduction to the Fake News Challenge Dataset

The **Fake News Challenge (FNC-1)** dataset was created to support the development of machine learning models for **stance detection** — a subtask of fake news detection. The challenge focuses on determining the relationship between a **news headline** and the corresponding **news article body**. By identifying how a headline and an article relate (e.g., whether they agree, disagree, or are unrelated), this dataset enables the development of tools to assist journalists and researchers in identifying potentially misleading or fake news.

### 🎯 Objective

The core objective is to classify the **stance** of a news article body with respect to its headline. This helps assess whether headlines are misleading or consistent with the article content.

---

## 📁 Dataset Files

The dataset consists of two CSV files:

### 1. **train\_bodies.csv**

This file contains the **full body text of news articles**. Each article body is identified by a unique `Body ID`. It serves as a reference for linking article content to corresponding headlines.

**Columns:**

* **Body ID**: A unique identifier for each article body.
* **articleBody**: The full text of the news article.

---

### 2. **train\_stances.csv**

This file contains labeled examples that pair **news headlines** with article bodies. Each row represents a (headline, body) pair along with the corresponding **stance** label.

**Columns:**

* **Headline**: A short news headline or title.
* **Body ID**: Refers to an entry in `train_bodies.csv`, linking the headline to the associated article body.
* **Stance**: The labeled relationship between the headline and the article body. It can take one of the following four values:

  * **agree**: The article supports or agrees with the claim made in the headline.
  * **disagree**: The article opposes or contradicts the headline.
  * **discuss**: The article discusses the same topic as the headline but does not clearly agree or disagree.
  * **unrelated**: The article is not about the same topic as the headline.

---

## 📊 Class Distribution

The stance labels are imbalanced, meaning some classes appear more frequently than others. The approximate distribution in the dataset (from `train_stances.csv`) is as follows:

| Stance    | Proportion |
| --------- | ---------- |
| Unrelated | 73.1%      |
| Discuss   | 17.8%      |
| Agree     | 7.4%       |
| Disagree  | 1.7%       |

This imbalance highlights the need for appropriate strategies (e.g., class weighting or resampling) when training classification models.

---

## 🧠 Use Case

This dataset is ideal for training and evaluating models that perform **stance detection**, a task crucial for applications in:

* Fake news detection
* Media monitoring
* Fact-checking systems
* Automated journalism tools

---
