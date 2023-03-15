import math


class State:
    def __init__(self, features_dict, exclude_features=None):
        self.features_dict = features_dict
        if exclude_features:
            for feature in exclude_features:
                if feature in self.features_dict:
                    self.features_dict.pop(feature)

    def distance(self, state):
        distance = 0
        for key, feature in self.feautres_dict.items():
            distance += (state.features_dict[key] - feature) ** 2

        return math.sqrt(distance)

    def feature_difference(self, state):
        difference_dict = {}
        for key, feature in self.features_dict.items():
            difference_dict[key] = feature - state.features_dict[key]

        return difference_dict

    def percentage_difference(self, state):
        percentage_dict = {}
        for key, feature in self.features_dict.items():
            difference = feature - state.features_dict[key]
            if abs(state.features_dict[key]) < 0.0001:
                percentage_dict[key] = 'infinite'
            else:
                percentage_dict[key] = 100 * (difference) / abs(state.features_dict[key])
        return percentage_dict

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.features_dict == other.features_dict
        else:
            return False