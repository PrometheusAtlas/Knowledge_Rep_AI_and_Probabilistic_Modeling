# Project: Knowledge Representation and Probabilistic Reasoning in AI

Implements an integrated artificial intelligence framework combining **First-Order Logic (FOL) inference**, **Bayesian Network modeling**, and **Naïve Bayes learning**, bridging symbolic and probabilistic reasoning under the **AIMA Python** environment.  
The project demonstrates unified representation, inference, and learning across declarative logic and uncertainty-based models.

---

## Part 1 – First-Order Logic Knowledge Base and Inference

Constructed a declarative knowledge base using Horn clauses and logical entailment for reasoning over structured relational domains.  
Implemented both **forward** and **backward chaining** inference strategies to perform logical deduction, unification, and variable substitution across complex rule sets.

### Technical Focus
- Defined recursive Horn clauses for hierarchical relationships and transitive reasoning.  
- Implemented inference using **AIMA’s FolKB**, `fol_fc_ask`, and `fol_bc_ask` modules.  
- Compared reasoning efficiency and completeness between chaining approaches.  
- Evaluated the scalability of inference across increasing rule base sizes.  

---

## Part 2 – Bayesian Network Modeling and Probabilistic Inference

Developed a **Bayesian Network** capturing dependencies between socioeconomic and environmental variables such as innovation, education, and sustainability metrics.  
Performed both **exact inference** (enumeration, variable elimination) and **approximate inference** (prior sampling, likelihood weighting, Gibbs sampling) to evaluate posterior distributions and probabilistic dependencies.

### Technical Focus
- Designed and parameterized **conditional probability tables (CPTs)** for all nodes.  
- Verified conditional independence and causal consistency of the network structure.  
- Quantified marginal and conditional probabilities under multiple evidence scenarios.  
- Visualized probabilistic dependencies using **NetworkX** and **Matplotlib** DAG plots.  

---

## Part 3 – Naïve Bayes Classification and Data-Driven Reasoning

Implemented **supervised probabilistic learning** using **Naïve Bayes** classifiers on real-world datasets (Iris, Maternal Health Risk).  
Extended the AIMA Naïve Bayes framework with customized data preprocessing, imputation, and posterior-distribution visualization.

### Technical Focus
- Integrated data handling and imputation using **Pandas** and **NumPy**.  
- Trained models via maximum-likelihood estimation with Laplace smoothing.  
- Evaluated accuracy, confusion matrices, and posterior reliability across datasets.  
- Produced interpretable probabilistic visualizations with **Seaborn** and **Matplotlib**.  

---

## Technical Overview

- **Frameworks:** AIMA Python, Pandas, NumPy, Matplotlib, Seaborn, NetworkX, Scikit-learn  
- **Core Methods:** First-Order Logic inference · Bayesian Networks · Probabilistic Learning  
- **Conceptual Scope:** Logical reasoning · Probabilistic inference · Uncertainty modeling · Explainable AI  

Developed to demonstrate a complete hybrid reasoning architecture — integrating symbolic logic, probabilistic graphical modeling, and machine-learning-based classification within a unified experimental platform.
