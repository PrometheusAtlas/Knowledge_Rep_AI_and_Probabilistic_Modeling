
import os
import sys
#sys.path.append('/home/t_hurl/aima-python')

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from logic import FolKB, fol_fc_ask, fol_bc_ask, expr
from probability import BayesNet, enumeration_ask, elimination_ask, gibbs_ask, likelihood_weighting, prior_sample, rejection_sampling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from learning import NaiveBayesLearner, DataSet
from sklearn.preprocessing import LabelEncoder



def FOL_family():

    # Step 1: Define the knowledge base in FOL
    kb = FolKB()

    # Step 2: Define rules in FOL

    # Ancestor relationships
    kb.tell(expr('Parent(x, y) ==> Ancestor(x, y)'))
    kb.tell(expr('(Parent(x, z) & Ancestor(z, y)) ==> Ancestor(x, y)'))

    # Sibling relationships
    kb.tell(expr('(Parent(p, x) & Parent(p, y) & Distinct(x, y)) ==> Sibling(x, y)'))

    # Cousin relationships
    kb.tell(expr('(Parent(p1, x) & Parent(p2, y) & Sibling(p1, p2) & Distinct(x, y)) ==> Cousin(x, y)'))

    # Inheritance
    kb.tell(expr('(Parent(x, y) & HasBlueEyes(x)) ==> MayHaveBlueEyes(y)'))

    # Step 3: Add the given facts
    kb.tell(expr('Parent(Alice, Carol)'))
    kb.tell(expr('Parent(Bob, Carol)'))
    kb.tell(expr('Parent(Alice, Dave)'))
    kb.tell(expr('Parent(Bob, Dave)'))
    kb.tell(expr('Parent(Carol, Frank)'))
    kb.tell(expr('HasBlueEyes(Carol)'))
    kb.tell(expr('Spouse(Dave, Eve)'))
    kb.tell(expr('Spouse(Eve, Dave)'))

    # Define distinct individuals
    kb.tell(expr('Distinct(Alice, Bob, Carol, Dave, Eve, Frank)'))

    # Step 4: Queries

    # Query 1: Does Frank have blue eyes?
    print("\nQuery 1: Does Frank possibly have blue eyes?")
    print("Using Forward Chaining:")
    fc_result = fol_fc_ask(kb, expr('MayHaveBlueEyes(Frank)'))
    print("Result:", list(fc_result))

    print("\nUsing Backward Chaining:")
    bc_result = fol_bc_ask(kb, expr('MayHaveBlueEyes(Frank)'))
    print("Result:", list(bc_result))

    # Query 2: Does Frank have an ancestor with blue eyes?
    print("\nQuery 2: Does Frank have an ancestor with blue eyes?")
    
    # First, find all ancestors of Frank
    ancestors_fc = fol_fc_ask(kb, expr('Ancestor(v, Frank)'))
    ancestors_bc = fol_bc_ask(kb, expr('Ancestor(v, Frank)'))

    # Now, check which ancestors have blue eyes
    print("Using Forward Chaining:")
    fc_results = []
    for ans in ancestors_fc:
        for var, val in ans.items():
            # Check if this ancestor has blue eyes
            if list(fol_fc_ask(kb, expr(f'HasBlueEyes({val})'))):
                fc_results.append({var: val})
    print("Result:", fc_results)

    print("\nUsing Backward Chaining:")
    bc_results = []
    for ans in ancestors_bc:
        for var, val in ans.items():
            # Check if this ancestor has blue eyes
            if list(fol_bc_ask(kb, expr(f'HasBlueEyes({val})'))):
                bc_results.append({var: val})
    print("Result:", bc_results)

    # Query 3: Are Carol and Eve cousins?
    print("\nQuery 3: Are Carol and Eve cousins?")
    fc_result = fol_fc_ask(kb, expr('Cousin(Carol, Eve)'))
    print("Using Forward Chaining:")
    print("Result:", list(fc_result))

    bc_result = fol_bc_ask(kb, expr('Cousin(Carol, Eve)'))
    print("Using Backward Chaining:")
    print("Result:", list(bc_result))

    # Query 4: All possible ancestor relationships
    print("\nQuery 4: All possible ancestor relationships.")
    fc_result = fol_fc_ask(kb, expr('Ancestor(x, y)'))
    print("Using Forward Chaining:")
    for res in fc_result:
        print(dict(res))

    bc_result = fol_bc_ask(kb, expr('Ancestor(x, y)'))
    print("Using Backward Chaining:")
    for res in bc_result:
        print(dict(res))


FOL_family()

#1.2 BAYESIAN NETWORKS

# Bayesian Network definition
impact_network = BayesNet([
    ('Urbanisation', '', 0.6),  # Independent node
    ('TechInnovation', '', 0.5),  # Independent node
    ('Education', 'TechInnovation', {True: 0.8, False: 0.3}),  # Education depends on TechInnovation
    ('JobMarket', 'Education', {True: 0.9, False: 0.4}),  # JobMarket depends on Education
    ('CleanEnergyAdoption', 'TechInnovation', {True: 0.7, False: 0.3}),  # CleanEnergyAdoption depends on TechInnovation
    ('ElectricCarPurchases', 'Urbanisation CleanEnergyAdoption', {  # Depends on Urbanisation and CleanEnergyAdoption
        (True, True): 0.8,
        (True, False): 0.4,
        (False, True): 0.6,
        (False, False): 0.2
    }),
    ('CarbonEmissions', 'ElectricCarPurchases CleanEnergyAdoption', {  # Depends on ElectricCarPurchases and CleanEnergyAdoption
        (True, True): 0.2,
        (True, False): 0.4,
        (False, True): 0.5,
        (False, False): 0.8
    }),
    ('AirQuality', 'Urbanisation', {True: 0.3, False: 0.7}),  # Depends on Urbanisation
    ('EcologicalFootprint', 'CarbonEmissions AirQuality', {  # Depends on CarbonEmissions and AirQuality
        (True, True): 0.85,
        (True, False): 0.7,
        (False, True): 0.4,
        (False, False): 0.1
    })
])

# Position the nodes etc
fixed_positions = {
    'Urbanisation': (0, 2),
    'AirQuality': (2, 2),
    'ElectricCarPurchases': (0, 0),
    'CarbonEmissions': (2, 0),
    'EcologicalFootprint': (4, 0),
    'CleanEnergyAdoption': (0, -2),
    'TechInnovation': (-3, 0), 
    'Education': (-1, 0),       
    'JobMarket': (-1, 2)
}

# Draw the Bayes Network
def draw_bayes_net():
    graph = nx.DiGraph()
    edges = [
        ('Urbanisation', 'ElectricCarPurchases'),
        ('TechInnovation', 'CleanEnergyAdoption'),
        ('TechInnovation', 'Education'),
        ('Education', 'JobMarket'),
        ('CleanEnergyAdoption', 'ElectricCarPurchases'),
        ('ElectricCarPurchases', 'CarbonEmissions'),
        ('CleanEnergyAdoption', 'CarbonEmissions'),
        ('Urbanisation', 'AirQuality'),
        ('CarbonEmissions', 'EcologicalFootprint'),
        ('AirQuality', 'EcologicalFootprint')
    ]
    graph.add_edges_from(edges)

    # Edge labels
    edge_labels = {
        ('Urbanisation', 'ElectricCarPurchases'): "Urban -> EV",
        ('TechInnovation', 'CleanEnergyAdoption'): "Tech -> Clean Energy",
        ('TechInnovation', 'Education'): "Tech -> Education",
        ('Education', 'JobMarket'): "Education -> Job Market",
        ('CleanEnergyAdoption', 'ElectricCarPurchases'): "Clean Energy -> EV",
        ('ElectricCarPurchases', 'CarbonEmissions'): "EV -> Carbon",
        ('CleanEnergyAdoption', 'CarbonEmissions'): "Clean Energy -> Carbon",
        ('Urbanisation', 'AirQuality'): "Urban -> Air",
        ('CarbonEmissions', 'EcologicalFootprint'): "Carbon -> Eco",
        ('AirQuality', 'EcologicalFootprint'): "Air -> Eco"
    }

    plt.figure(figsize=(12, 8))
    nx.draw(
        graph, pos=fixed_positions, with_labels=True,
        node_size=3000, node_color="lightblue", font_size=10, font_weight="bold"
    )
    nx.draw_networkx_edge_labels(
        graph, pos=fixed_positions, edge_labels=edge_labels, font_size=8
    )
    plt.title("Bayesian Network: Impact Assessment")
    plt.show()

# Sampling the network

def perform_sampling():
    print("\nPerforming Prior Sampling:")
    samples = [prior_sample(impact_network) for _ in range(1000)]
    print(f"Generated {len(samples)} samples.")
    print(f"Sampled Evidence: {samples[0]}")

    print("\nPerforming Rejection Sampling for P(CleanEnergyAdoption=True | TechInnovation=True):")
    rejection_result = rejection_sampling('CleanEnergyAdoption', {'TechInnovation': True}, impact_network, 1000)
    print(rejection_result.show_approx())

    print("\nPerforming Likelihood Weighting for P(ElectricCarPurchases | Urbanisation=True, CleanEnergyAdoption=True):")
    likelihood_result = likelihood_weighting('ElectricCarPurchases', {'Urbanisation': True, 'CleanEnergyAdoption': True}, impact_network, 1000)
    print(likelihood_result.show_approx())

    print("\nPerforming Gibbs Sampling for P(EcologicalFootprint | CarbonEmissions=True, AirQuality=True):")
    gibbs_result = gibbs_ask('EcologicalFootprint', {'CarbonEmissions': True, 'AirQuality': True}, impact_network, 1000)
    print(gibbs_result.show_approx())

# Query the network
def query_impact_network():
    print("\nQuerying the Bayesian Network:\n")

    # Exact inference method: Enumeration
    print("Exact Inference (Enumeration) - P(CarbonEmissions | CleanEnergyAdoption=True):")
    print(enumeration_ask('CarbonEmissions', {'CleanEnergyAdoption': True}, impact_network).show_approx())

    # Exact inference method: Variable Elimination
    print("\nExact Inference (Variable Elimination) - P(JobMarket | Education=True):")
    print(elimination_ask('JobMarket', {'Education': True}, impact_network).show_approx())


# Visualize
draw_bayes_net()

# Run the sampling
perform_sampling()

# Execute all queries
query_impact_network()


#Question 1.3.1


    #Part 1: dataset 1 - the Iris dataset (https://archive.ics.uci.edu/dataset/53/iris)


# Load the Iris dataset

iris_path = "/Data/iris.data"
iris_columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
iris_data = pd.read_csv(iris_path, header=None, names=iris_columns)

# Prior Probabilities (P(A))
prior_probs = iris_data["Species"].value_counts(normalize=True)
print("Iris Dataset - Prior Probabilities (P(A)):")
#print(prior_probs)

# Evidence Probabilities (P(B))
evidence_probs = {}
for feature in iris_columns[:-1]:  # Exclude 'Species'
    evidence_probs[feature] = iris_data[feature].value_counts(normalize=True)
print("\nIris Dataset - Evidence Probabilities (P(B)):")
#for feature, probs in evidence_probs.items():
#    print(f"{feature}:")
#    print(probs)

# Likelihoods (P(B|A))
likelihoods = {}
for feature in iris_columns[:-1]: 
    likelihoods[feature] = iris_data.groupby("Species")[feature].value_counts(normalize=True)
print("\nIris Dataset - Likelihoods (P(B|A)):")
#for feature, probs in likelihoods.items():
#    print(f"{feature}:")
#    print(probs)

# Posterior (P(A|B)) using Bayes Theorem 
posteriors = {}
for feature in iris_columns[:-1]: 
    posteriors[feature] = {}
    for species in prior_probs.index:
        if species in likelihoods[feature].index:
            posterior = (likelihoods[feature].loc[species] * prior_probs[species]) / evidence_probs[feature]
            posteriors[feature][species] = posterior.mean()

# Creates a dependency graph
G = nx.DiGraph()
G.add_node("Species", size=3000, color="red")

# Add feature nodes and annotate edges with likelihood and posterior
edge_labels = {}
for feature in iris_columns[:-1]:
    G.add_node(feature, size=1500, color="skyblue")
    G.add_edge("Species", feature, weight=1.0)

    # Join posterior closer to species
    posterior_details = "\n".join(
        [f"P({species}|{feature}) ≈ {posteriors[feature][species]:.2f}"
         for species in posteriors[feature]]
    )

    # Likelihood closer to feature
    likelihood_label = f"P({feature}|Species)"

    # Update edge labels, use below in draw_networkx_edge
    edge_labels[("Species", feature)] = f"{posterior_details}\n{likelihood_label}"

# Plot the updated dependency graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)

# Draws nodes
node_colors = [G.nodes[node]["color"] for node in G.nodes]
node_sizes = [G.nodes[node]["size"] for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray")

# Annotates edges with likelihood and posterior prob
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Add labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
plt.title("Dependency Graph: Likelihood & Posteriors", fontsize=16)
plt.axis("off")
plt.show()


# Part 2: dataset 2 - the Maternal Health Risk dataset

health_path = "/Data/Maternal Health Risk Data Set.csv"
health_data = pd.read_csv(health_path)

# Prior Probabilities (P(A))
prior_probs = health_data["RiskLevel"].value_counts(normalize=True)
print("\nMaternal Health Risk Dataset - Prior Probabilities (P(A)):")
#print(prior_probs)

# Evidence Probabilities (P(B))
evidence_probs = {}
for feature in health_data.columns[:-1]:  # Exclude RiskLevel
    evidence_probs[feature] = health_data[feature].value_counts(normalize=True)
print("\nMaternal Health Risk Dataset - Evidence Probabilities (P(B)):")
#for feature, probs in evidence_probs.items():
#    print(f"{feature}:")
#    print(probs)

# Likelihoods (P(B|A))
likelihoods = {}
for feature in health_data.columns[:-1]: 
    likelihoods[feature] = health_data.groupby("RiskLevel")[feature].value_counts(normalize=True)
print("\nMaternal Health Risk Dataset - Likelihoods (P(B|A)):")
#for feature, probs in likelihoods.items():
#    print(f"{feature}:")
#    print(probs)

# Posterior Probabilities (P(A|B)) using Bayes Theorem
posteriors = {}
for feature in health_data.columns[:-1]: 
    posteriors[feature] = {}
    for risk_level in prior_probs.index:
        if risk_level in likelihoods[feature].index:
            posterior = (likelihoods[feature].loc[risk_level] * prior_probs[risk_level]) / evidence_probs[feature]
            posteriors[feature][risk_level] = posterior.mean()

# Creates a dependency graph
G = nx.DiGraph()
G.add_node("RiskLevel", size=3000, color="red")

# Add feature nodes and annotate edges with likelihood and posterior
edge_labels = {}
for feature in health_data.columns[:-1]:  # Exclude 'RiskLevel'
    G.add_node(feature, size=1500, color="skyblue")
    G.add_edge("RiskLevel", feature, weight=1.0)

    # Joins posterior closer to Risk Level
    posterior_details = "\n".join(
        [f"P({risk_level}|{feature}) ≈ {posteriors[feature][risk_level]:.2f}"
         for risk_level in posteriors[feature]]
    )

    # Likelihood closer to feature
    likelihood_label = f"P({feature}|RiskLevel)"
    
    # Update edge labels, use below in draw_networkx_edge
    edge_labels[("RiskLevel", feature)] = f"{posterior_details}\n{likelihood_label}"

# Visualize the updated Dependency Graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)

# Draw nodes
node_colors = [G.nodes[node]["color"] for node in G.nodes]
node_sizes = [G.nodes[node]["size"] for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray")

# Annotate edges with switched likelihood and posterior labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Add labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
plt.title("Dependency Graph: Likelihood & Posteriors", fontsize=16)
plt.axis("off")
plt.show()



# Naive Bayes Classification - Datasets: part 2. 


def imputation_for_missing_values(self):
    """
    Debug function to calculate means and standard deviations.
    Provides warnings for mismatched feature counts and imputes missing values with class-specific means.
    """
    from statistics import mean, stdev
    target_names = self.values[self.target]
    feature_numbers = len(self.inputs)

    item_buckets = self.split_values_by_classes()  # Group items by class label

    means = {t: [0] * feature_numbers for t in target_names}
    deviations = {t: [0] * feature_numbers for t in target_names}

    for t in target_names:
        features = [[] for _ in range(feature_numbers)]  # Initialize feature buckets
        for item in item_buckets[t]:
            if len(item) != feature_numbers:  # Warn and replace rows with missing values
                print(f"Warning: Item {item} has {len(item)} features instead of {feature_numbers}. Imputing.")
                while len(item) < feature_numbers:
                    item.append(0)  # Add placeholder zeros for missing features, ensures the process can finish. 
            for i in range(feature_numbers):
                features[i].append(item[i])  # Add feature values by index

        # Calculate means and standard deviations
        for i in range(feature_numbers):
            if len(features[i]) > 1:
                means[t][i] = mean(features[i])
                deviations[t][i] = stdev(features[i])
            else:
                means[t][i] = mean(features[i]) if features[i] else 0
                deviations[t][i] = 0

    # Impute missing values
    corrected_items = []  # Track corrected rows
    for t in target_names:
        for item in item_buckets[t]:
            for i in range(feature_numbers):
                 # Replaces the temporary zeros with mean imputation. 
                if item[i] == 0: 
                    print("\nImputed and Corrected Rows:")
                    item[i] = means[t][i]
            corrected_items.append(item)



    return means, deviations

DataSet.find_means_and_deviations = imputation_for_missing_values
# Load Maternal Health Risk Dataset
print("\nLoading and Preprocessing Dataset\n")
health_data_ = pd.read_csv('/Data/Maternal Health Risk Data Set.csv')

# Basic dataset exploration
print("Dataset Info:")
print(health_data_.info())
print("\nMissing Values:")
print(health_data_.isnull().sum())

# Visualize relationships
sns.pairplot(health_data_, diag_kind='kde', hue='RiskLevel')
plt.show()

# Define features and target
feature_names_ = health_data_.columns[:-1].tolist()
target_name_ = 'RiskLevel'

# Encode target labels from continuous to discrete
label_encoder_ = LabelEncoder()
health_data_[target_name_] = label_encoder_.fit_transform(health_data_[target_name_])
target_classes_ = label_encoder_.classes_

# Split data into train-test sets
X_ = health_data_[feature_names_].values
y_ = health_data_[target_name_].values
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2, random_state=42)

# Create DataFrames for train and test data
train_df_ = pd.DataFrame(X_train_, columns=feature_names_)
train_df_['target'] = y_train_
test_df_ = pd.DataFrame(X_test_, columns=feature_names_)
test_df_['target'] = y_test_

# Prepare data for AIMA
train_data_ = train_df_.values.tolist()
test_data_ = test_df_.values.tolist()

# Ensure all examples have the correct number of attributes
expected_length_ = len(feature_names_) + 1  # Features + target
train_data_ = [row for row in train_data_ if len(row) == expected_length_]
test_data_ = [row for row in test_data_ if len(row) == expected_length_]

print(f"\nFiltered training examples: {len(train_data_)}")
print(f"Filtered testing examples: {len(test_data_)}")

# Train NaiveBayesLearner
print("\nTraining Naive Bayes Learner\n")
nb_learner_health_ = NaiveBayesLearner(DataSet(examples=train_data_, target=-1))

# Evaluate the model
print("\nEvaluating the Classifier\n")
predictions_ = [nb_learner_health_(row[:-1]) for row in test_data_]
actuals_ = [row[-1] for row in test_data_]

# Metrics
accuracy_ = accuracy_score(actuals_, predictions_)
report_ = classification_report(actuals_, predictions_, target_names=target_classes_)
print(f"Accuracy: {accuracy_}")
print("Classification Report:\n", report_)

cm_ = confusion_matrix(actuals_, predictions_, labels=range(len(target_classes_)))
disp_ = ConfusionMatrixDisplay(confusion_matrix=cm_, display_labels=target_classes_)
disp_.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Maternal Health Risk Dataset")
plt.show()

#------------------------------------------------------------------------------------

# Load the Iris dataset
iris_path = "/Data/iris.data"
iris_columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
iris_data = pd.read_csv(iris_path, header=None, names=iris_columns)

# Basic dataset exploration
#print("Dataset Info:")
#print(iris_data.info())
#print("\nMissing Values:")
#print(iris_data.isnull().sum())

# Visualize relationships
sns.pairplot(iris_data, diag_kind='kde', hue='Species')
plt.show()

# Define features and target
feature_names = iris_columns[:-1]
target_name = "Species"

# Encode target labels from continuous to discrete
label_encoder = LabelEncoder()
iris_data[target_name] = label_encoder.fit_transform(iris_data[target_name])
target_classes = label_encoder.classes_

# Split data into train-test sets
X = iris_data[feature_names].values
y = iris_data[target_name].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataFrames for train and test data
train_df = pd.DataFrame(X_train, columns=feature_names)
train_df['target'] = y_train
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['target'] = y_test

# Prepare data for AIMA
train_data = train_df.values.tolist()
test_data = test_df.values.tolist()

# Ensure all examples have the correct number of attributes
expected_length = len(feature_names) + 1  # Features + target
train_data = [row for row in train_data if len(row) == expected_length]
test_data = [row for row in test_data if len(row) == expected_length]

print(f"\nFiltered training examples: {len(train_data)}")
print(f"Filtered testing examples: {len(test_data)}")

# Train NaiveBayesLearner
print("\nTraining Naive Bayes Learner\n")
nb_learner_iris = NaiveBayesLearner(DataSet(examples=train_data, target=-1))


# Evaluate the model
print("\nEvaluating the Classifier\n")
predictions = [nb_learner_iris(row[:-1]) for row in test_data]
actuals = [row[-1] for row in test_data]

# Metrics
accuracy = accuracy_score(actuals, predictions)
report = classification_report(actuals, predictions, target_names=target_classes)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

cm = confusion_matrix(actuals, predictions, labels=range(len(target_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Iris Dataset")
plt.show()


