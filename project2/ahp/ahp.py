import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np


# tree funcs
def plot_tree(tree_df):
    G = nx.DiGraph()
    edges = [(row["parent"], row["child"]) for idx, row in tree_df.iterrows()]
    G.add_edges_from(edges)
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()


def plot_tree_with_weights(tree_df, weights):
    G = nx.DiGraph()
    edges = [(row["parent"], row["child"]) for idx, row in tree_df.iterrows()]
    G.add_edges_from(edges)
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=False, arrows=True, font_size=1)

    labels = {}
    for node, (x, y) in pos.items():
        weight = weights.get(node + "_comparison")
        weight = np.delete(weight, np.where(weight == None))
        weight = np.round(weight, 2)
        labels[node] = f"{node}\n\n{weight}" if len(weight) > 0 else node

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        horizontalalignment="center",
        verticalalignment="top",
        font_size=10,
    )
    plt.show()


def generate_placeholder_comparison_matrices(tree_structure_df):
    comparison_matrices = {}
    for parent_node in tree_structure_df["parent"].unique():
        children = tree_structure_df[tree_structure_df["parent"] == parent_node][
            "child"
        ].tolist()

        matrix = pd.DataFrame(index=children, columns=children, dtype=np.float64)

        for i in range(len(children)):
            for j in range(len(children)):
                matrix.iloc[i, j] = 1.0

        comparison_matrices[parent_node] = matrix
    return comparison_matrices


# matrix func
def calculate_consistency_ratio(matrix):
    def calculate_relative_consistency_index(matrix):
        n = len(matrix)
        lambda_max = np.linalg.eig(matrix.values)[0].max()
        consistency_index = (lambda_max - n) / (n - 1)
        return consistency_index

    random_indices = {
        1: 0.0001,
        2: 0.0001,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
        11: 1.51,
    }
    n = len(matrix)
    consistency_index = calculate_relative_consistency_index(matrix)
    random_index = random_indices[n]
    consistency_ratio = consistency_index / random_index
    return consistency_ratio


def show_comparison_matrices(comparison_matrices):
    for node, matrix in comparison_matrices.items():
        print(f"Comparison Matrix for children of {node}:")
        print(matrix)
        print()


def save_comparison_matrices(comparison_matrices, output_dir):
    for node, matrix in comparison_matrices.items():
        matrix.to_csv(
            os.path.join(output_dir, f"{node}_comparison_matrix.csv"), index=False
        )


def load_comparison_matrices(directory):
    comparison_matrices_data = {}
    files = [
        file
        for file in os.listdir(directory)
        if file.endswith("_comparison_matrix.csv")
    ]

    for file in files:
        node = "_".join(file.split("_")[:-1])
        if node[0].isupper():
            continue
        df = pd.read_csv(os.path.join(directory, file))
        comparison_matrices_data[node] = df

    return comparison_matrices_data


def initialize_comparison_matrices(tree_structure_df, output_dir):
    comparison_matrices = generate_placeholder_comparison_matrices(tree_structure_df)
    save_comparison_matrices(comparison_matrices, output_dir)


# eigenvector
def approximate_principal_eigenvector(matrix):
    initial = np.sum(matrix, axis=1)
    initial_weights = initial / np.sum(initial)
    while True:
        matrix = matrix @ matrix
        rows = np.sum(matrix, axis=1)
        weights = rows / np.sum(rows)
        if np.allclose(weights, initial_weights, atol=1e-10):
            break
        initial_weights = weights
    return weights


def recreate_matrix_from_weights(weights):
    n = len(weights)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = weights[i] / weights[j]
            matrix[j][i] = 1 / matrix[i][j]
    np.fill_diagonal(matrix, 1)
    return matrix


# criteria perf
def calculate_normalized_performances(data_df):
    criteria = data_df.columns[1:]
    normalized_data = data_df.copy()
    for criterion in criteria:
        column_sum = normalized_data[criterion].sum()
        normalized_data[criterion] = normalized_data[criterion] / column_sum
    return normalized_data


# gather
def calculate_comprehensive_scores(
    normalized_data, criteria_weights, comparison_matrices, tree_df
):
    comprehensive_scores = {}
    low_level = normalized_data.columns[1:]
    for index, row in normalized_data.iterrows():
        score = 0
        row = row[1:]
        for i, criterion in enumerate(low_level):
            criterion_score = row[i]
            while True:
                parent_node = tree_df[tree_df["child"].str.match(criterion)]
                if parent_node.empty:
                    break
                parent_node = parent_node.iloc[0]["parent"]
                criteria_weights_name = f"{parent_node}_comparison"
                comparison_matrices_idx = (
                    comparison_matrices[criteria_weights_name]
                    .columns.tolist()
                    .index(criterion)
                )
                criterion_score *= criteria_weights[criteria_weights_name][
                    comparison_matrices_idx
                ]
                criterion = parent_node
            score += criterion_score
        comprehensive_scores[normalized_data.iloc[index, 0]] = score
    return comprehensive_scores


def rank_alternatives(comprehensive_scores):
    return sorted(comprehensive_scores.items(), key=lambda x: x[1], reverse=True)


def perform_ahp(data_df, comparison_matrices, tree_df):
    normalized_data = calculate_normalized_performances(data_df)
    criteria_weights = {}
    for node, matrix in comparison_matrices.items():
        criteria_weights[node] = approximate_principal_eigenvector(matrix.values)

    comprehensive_scores = calculate_comprehensive_scores(
        normalized_data, criteria_weights, comparison_matrices, tree_df
    )

    return rank_alternatives(comprehensive_scores)
