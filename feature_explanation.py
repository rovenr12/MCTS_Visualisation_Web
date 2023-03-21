import data_preprocessing
import math
import pandas as pd

DIFFERENCE_TYPE = ['Higher', 'Same', 'Lower']


def get_children_in_depth(df, exclude_action_names=None):
    """
    Generate the dictionary about the related nodes in different depths for each root available actions
    :param df: MCTS data file
    :param exclude_action_names: the list of root action names that will be ignored
    :return: dictionary about the related nodes in different depths for each root available actions,
             minimum depth of available actions, maximum depth of available actions
    """

    def helper(current_depth, root_action):
        """
        Recursive function that helps to generate the list of children node for particular
        root action in different depth
        :param current_depth: current depth of root action
        :param root_action: the name of root action
        """
        if current_depth not in root_actions_depth_dict[root_action]:
            return

        # Generate the list of all available node's name from current depth
        children_actions_from_current_depth_list = []
        for children_action in root_actions_depth_dict[root_action][current_depth]:
            children_actions_from_current_depth_list += data_preprocessing.get_node_available_actions(df,
                                                                                                      children_action)

        # Add to dictionary if the list is not empty
        if len(children_actions_from_current_depth_list) != 0:
            root_actions_depth_dict[root_action][current_depth + 1] = children_actions_from_current_depth_list

        # Go to next depth level
        return helper(current_depth + 1, root_action)

    root_actions = data_preprocessing.get_root_available_actions(df, exclude_action_names)
    root_actions_depth_dict = {action: {0: [action]} for action in root_actions}

    max_depth = float('-inf')
    min_depth = float('inf')

    # Generate the depth dictionary for different root actions and
    # record the minimum and maximum depth for the MCTS tree
    for action in root_actions:
        helper(0, action)
        action_max_depth = max(root_actions_depth_dict[action])
        max_depth = max(max_depth, action_max_depth)
        min_depth = min(min_depth, action_max_depth)

    return root_actions_depth_dict, min_depth, max_depth


def get_feature_counts(df, root_features, node_list, exclude_features=None, feature_col='Game_Features', rel_tol=0.001):
    """
    Return the game feature summary about feature changes between root features and other features in node_list
    :param df: MCTS data file
    :param root_features: the root node features
    :param node_list: the list of node names
    :param exclude_features: the list of features will be ignored
    :param feature_col: the column name of features
    :param rel_tol: the column name used for getting available actions (It needs to be unique for different nodes)
    :return: game feature summary dictionary
    """
    # Init the game feature summary dictionary
    game_feature_summary = {feature: {difference_type: 0 for difference_type in DIFFERENCE_TYPE} for feature in
                            root_features}

    # count the features changes for every feature in every node
    for node_name in node_list:
        node_features = data_preprocessing.get_features(df, node_name, exclude_features, feature_col)
        for feature in root_features:
            root_val = root_features[feature]
            node_val = node_features[feature]

            if math.isclose(root_val, node_val, rel_tol=rel_tol):
                game_feature_summary[feature]['Same'] += 1
            elif node_val > root_val:
                game_feature_summary[feature]['Higher'] += 1
            else:
                game_feature_summary[feature]['Lower'] += 1

    return game_feature_summary


def generate_game_feature_explanation_df(df, depth_type='max', exclude_action_nodes=None, exclude_features=None,
                                         feature_col='Game_Features', rel_tol=0.001):
    """
    Return the dataframe of explanations based on game features changes in different depth
    :param df: MCTS data file
    :param depth_type: can be ['max', 'min', 'average']. It will decide the depth of each feature. Max means using the
                        maximum depth of root actions. Min means using the minimum depth of root actions. Average means
                        taking the average of maximum depth and minimum depth
    :param exclude_action_nodes: the list of root actions will be ignored
    :param exclude_features: the list of features will be ignored
    :param feature_col: the column name of features
    :param rel_tol: the column name used for getting available actions (It needs to be unique for different nodes)
    :return: the game features explanation dataframe
    """
    # Generate the dictionary about the related nodes in different depths for each root available actions
    root_actions_depth_dict, min_depth, max_depth = get_children_in_depth(df, exclude_action_nodes)

    # Decide the maximum depth for the dataframe
    if depth_type == 'max':
        maximum_depth = max_depth
    elif depth_type == 'min':
        maximum_depth = min_depth
    else:
        maximum_depth = (max_depth + min_depth) // 2

    # Get the root node features
    root_name = data_preprocessing.get_root_node_name(df)
    root_features = data_preprocessing.get_features(df, root_name, exclude_features, feature_col)

    features_differences = {}

    # Loop for each available root action and do to feature change statistics
    for root_action, action_depth_dict in root_actions_depth_dict.items():
        root_action_name = data_preprocessing.node_name_to_action_name(df, root_action)
        # Create dictionaries for each different type
        for difference_type in DIFFERENCE_TYPE:
            features_differences[(root_action_name, difference_type)] = dict()

        # Loop through the different depth and summary the feature difference
        for depth, node_list in action_depth_dict.items():
            if depth > maximum_depth:
                break

            if depth == 0:
                depth = "Immediate"

            game_feature_summary = get_feature_counts(df, root_features, node_list, exclude_features, feature_col,
                                                      rel_tol)

            for feature, summary in game_feature_summary.items():
                for difference_type, count in summary.items():
                    features_differences[(root_action_name, difference_type)][
                        (feature, depth)] = f"{count / len(node_list) * 100:.1f}%({count})"

    # Change dictionary to Dataframe and reorder the index
    feature_order_dict = {name: idx for idx, name in enumerate(list(root_features))}
    feature_df = pd.DataFrame.from_dict(features_differences)
    feature_df.sort_index(inplace=True, key=lambda x: [i if type(i) == int else 0 for i in x], level=1)
    feature_df.sort_index(inplace=True, key=lambda x: [feature_order_dict[i] for i in x], level=0, sort_remaining=False)
    feature_df.index.names = ['Feature', 'Depth']

    return feature_df, maximum_depth
