import json


def get_node(df, node_name):
    """
    Get the specific node column from MCTS data file
    :param df: MCTS data file
    :param node_name: the name of node
    :return: Series
    """
    return df[df['Name'] == node_name].iloc[0]


def get_root_node(df):
    """
    Get the root node column from MCTS data file
    :param df: MCTS data file
    :return: Root node Series
    """
    return df[df['Parent_Name'] == 'None'].iloc[0]


def get_root_node_name(df):
    """
    Get the root node name from MCTS data file
    :param df: MCTS data file
    :return: Tree root name
    """
    return get_root_node(df)['Name']


def get_node_available_actions(df, node_name, exclude_actions=None, ref_col='Name'):
    """
    Return the list of available actions from certain node. It will exclude actions provided in excluded_actions
    if provided.
    :param df: MCTS data file
    :param node_name: the node name (need to matched its children Parent Name column)
    :param exclude_actions: the list of actions will be ignored
    :param ref_col: the column name used for getting available actions (It needs to be unique for different nodes)
    :return: the list of actions
    """
    actions = df[df['Parent_Name'] == node_name]

    if exclude_actions:
        actions = actions[~actions[ref_col].isin(exclude_actions)]

    return list(actions[ref_col].values)


def get_root_available_actions(df, exclude_actions=None, ref_col='Name'):
    """
    Return the list of available actions from root node. It will exclude actions provided in excluded_actions
    if provided.
    :param df: MCTS data file
    :param exclude_actions: the list of actions will be ignored
    :param ref_col: the column name used for getting available actions (It needs to be unique for different nodes)
    :return: the list of root actions
    """
    root_name = get_root_node_name(df)

    return get_node_available_actions(df, root_name, exclude_actions, ref_col)


def get_features(df, node_name, exclude_features=None, feature_col='Game_Features'):
    """
    Return the feature dictionary of a particular node
    :param df: MCTS data file
    :param node_name: the name of node
    :param exclude_features: the list of features that will be ignored
    :param feature_col: the name of feature column
    :return: features dictionary
    """
    node = get_node(df, node_name)
    game_features = json.loads(node[feature_col])

    if exclude_features:
        for exclude_feature in exclude_features:
            if exclude_feature in game_features:
                game_features.pop(exclude_feature)

    return game_features


def node_name_to_action_name(df, node_name):
    """
    Transfer the node name to action name
    :param df: MCTS data file
    :param node_name: the name of node
    :return: the action name
    """
    return df[df['Name'] == node_name].iloc[0]['Action_Name']

