import math


def square_rooted(x):
    return round(math.sqrt(sum([a * a for a in x])), 3)


def euclidean_distance(x, y):
    return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


distance_similarity_function_dict = {
    "Euclidean": euclidean_distance,
    "Manhattan": manhattan_distance,
    "Cosine": cosine_similarity
}


def jaccard(x, y):
    if not x and not y:
        return 1

    intersection = len(set(x) & set(y))
    union = len(x) + len(y) - intersection

    return intersection / union


def sorenson_dice(x, y):
    if not x and not y:
        return 1

    return 2 * len(set(x) & set(y)) / (len(x) + len(y))


def overlap_coefficient(x, y):
    if not x or not y:
        return 1

    return len(set(x) & set(y)) / (min(len(x), len(y)))


set_similarity_function_dict = {'Jaccard': jaccard,
                                'Sorenson-Dice': sorenson_dice,
                                'Overlap Coefficient': overlap_coefficient}
