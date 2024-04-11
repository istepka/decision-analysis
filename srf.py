def srf_procedure(rankings, Z):
    ranks = []
    running_whites = 0
    counter = 1
    for element in rankings[::-1]:
        if isinstance(element, list):
            ranks.append(counter + running_whites)
            counter += 1
        elif isinstance(element, int):
            running_whites += element
    max_rank = max(ranks)
    non_normalized_weights = [
        1 + (Z - 1) * (rank - 1) / (max_rank - 1) for rank in ranks
    ]
    weights_dict = {}
    counter = 0
    for rank in rankings[::-1]:
        if isinstance(rank, list):
            weights_dict.update({x: non_normalized_weights[counter] for x in rank})
            counter += 1

    sum_weights = sum(weights_dict.values())
    normalized_weights = {
        key: value / sum_weights for key, value in weights_dict.items()
    }
    return normalized_weights
