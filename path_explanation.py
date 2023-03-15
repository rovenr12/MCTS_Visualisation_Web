from state import State
import json


def get_action_path(df, node_name=None, action_name=None, is_max=True,
                    state_col='Game_Features', exclude_features=None):

    action_sequences = []
    root_node = df[df['Parent_Name'] == 'None'].iloc[0]
    root_name = root_node['Name']
    root_state = State(json.loads(root_node[state_col]), exclude_features)

    action_sequences.append(root_state)
    if node_name:
        filter_df = df[(df['Parent_Name'] == root_name) & (df['Name'] == node_name)]
    elif action_name:
        filter_df = df[(df['Parent_Name'] == root_name) & (df['Action_Name'] == action_name)]
    else:
        filter_df = df[df['Parent_Name'] == root_name]

    while filter_df.size:
        if is_max:
            current_row = filter_df.iloc[filter_df['Value'].argmax()]
        else:
            current_row = filter_df.iloc[filter_df['Value'].argmin()]
        current_state = State(json.loads(current_row[state_col]), exclude_features)
        current_name = current_row['Name']
        action_sequences.append(current_state)
        filter_df = df[df['Parent_Name'] == current_name]

    return action_sequences


def get_path_feature_difference(path, rel_tol=0.0001):
    feature_difference = path[-1].feature_difference(path[0])
    percentage_difference = path[-1].percentage_difference(path[0])
    feature_differences = {}
    percentage_differences = {}

    for name, difference in feature_difference.items():
        if abs(difference) > rel_tol:
            feature_differences[name] = difference
            percentage_differences[name] = percentage_difference[name]

    return feature_differences, percentage_differences


def generate_explanation_list(features_differences, percentage_differences,
                              features=None, percentage_differences_B=None):
    if features is None:
        features = features_differences.keys()

    feature_explanation = []
    addition = ""

    for feature in features:
        feature_difference = features_differences[feature]
        percentage_difference = percentage_differences[feature]
        if percentage_difference == 'infinite' or not percentage_differences_B:
            percentage_difference = percentage_difference
        else:
            percentage_difference = (percentage_difference + percentage_differences_B[feature]) / 2
            addition = 'around '

        if feature_difference > 0:
            if percentage_difference != 'infinite':
                feature_explanation.append(f"{feature} increase by {addition}{abs(percentage_difference):.1f}%")
            else:
                feature_explanation.append(f"{feature} increase")
        else:
            if percentage_difference != 'infinite':
                feature_explanation.append(f"{feature} decrease by {addition}{abs(percentage_difference):.1f}%")
            else:
                feature_explanation.append(f"{feature} decrease")

    return feature_explanation


def classify_features(features_A, features_B):
    common_features = set()
    for feature in features_A.keys() & features_B.keys():
        ba_differences = features_A[feature]
        wa_differences = features_B[feature]
        if ba_differences == 'infinite' or wa_differences == 'infinite':
            if ba_differences == wa_differences:
                common_features.add(feature)
        elif (ba_differences > 0 and wa_differences > 0) or (ba_differences < 0 and wa_differences < 0):
            common_features.add(feature)
    features_A_exclude_list = features_A.keys() - common_features
    features_B_exclude_list = features_B.keys() - common_features
    return common_features, features_A_exclude_list, features_B_exclude_list


def explanation_by_paths(df, state_col='Game_Features', exclude_features=None):
    explanations = []
    # Get the root node name and action space
    root_node_name = df[df['Parent_Name'] == 'None'].iloc[0].Name
    action_space = df[df.Parent_Name == root_node_name].shape[0]
    explanations.append(f"There are {action_space} actions available for this state.")

    # Check the value of all possible action, if the value is the same, it is just a random decision
    if len(df[df.Parent_Name == root_node_name]['Value'].unique()) == 1:
        explanations.append("The value of every action is the same.")

    # Get Best action name
    best_action_name = df[df['Parent_Name'] == 'None']['Best_Action'].values[0]

    # Get the path that will result in the highest reward and state difference between root node and last node
    best_action_paths = get_action_path(df, state_col=state_col, action_name=best_action_name, exclude_features=exclude_features)
    ba_feature_differences, ba_percentage_differences = get_path_feature_difference(best_action_paths)

    # Get the number of executed actions in the path
    actions_number = len(best_action_paths) - 1
    explanations.append(f"The explanation is based on the result of executing {actions_number} action(s) by the agent.")

    # Get the path that will result in the lowest reward and state difference between root node and last node
    worse_action_paths = get_action_path(df, action_name=best_action_name, is_max=False,
                                         exclude_features=exclude_features, state_col=state_col)
    worse_action_paths = worse_action_paths[:len(best_action_paths)]
    wa_feature_differences, wa_percentage_differences = get_path_feature_difference(worse_action_paths)

    # Generate the explanation
    # Situation 1: No difference between the best value path and the worst value path
    if ba_feature_differences == wa_feature_differences:
        feature_explanation = generate_explanation_list(ba_feature_differences, ba_percentage_differences)
        if len(feature_explanation) == 0:
            explanations.append(f"After executing {best_action_name}, nothing happens.")
            return explanations
        if len(feature_explanation) == 1:
            explanations.append(f"After executing {best_action_name}, I expected {feature_explanation[0]} in the future state.")
            return explanations
        explanations.append(f"After executing {best_action_name}, I expected {', '.join(feature_explanation[:-1])} and {feature_explanation[-1]} in the future state.")
        return explanations

    # Situation 2: Difference between the best value path and the worst value path
    # Classify the features
    common_features, ba_exclude_features, wa_exclude_features = classify_features(ba_feature_differences, wa_feature_differences)

    feature_explanation = generate_explanation_list(ba_feature_differences, ba_percentage_differences, common_features,
                                                    wa_percentage_differences)
    best_feature_explanation = generate_explanation_list(ba_feature_differences, ba_percentage_differences,
                                                         ba_exclude_features)
    worse_feature_explanation = generate_explanation_list(wa_feature_differences, wa_percentage_differences,
                                                          wa_exclude_features)

    # Combine explanations based on different situations
    if len(feature_explanation):
        union_sentence = f"{', '.join(feature_explanation[:-1])} and {feature_explanation[-1]}" if len(
            feature_explanation) > 1 else feature_explanation[0]
        explanations.append(f"After executing {best_action_name}, I expect {union_sentence} in the future state.")

    if len(best_feature_explanation):
        best_sentence = f"{', '.join(best_feature_explanation[:-1])} and {best_feature_explanation[-1]}" if len(
            best_feature_explanation) > 1 else best_feature_explanation[0]
        explanations.append(f"In the best situation, I also expect {best_sentence}.")

    if len(worse_feature_explanation):
        worst_sentence = f"{', '.join(worse_feature_explanation[:-1])} and {worse_feature_explanation[-1]}" if len(
            worse_feature_explanation) > 1 else worse_feature_explanation[0]
        explanations.append(f"However, I may expect {worst_sentence} in the worst situation.")

    return explanations


def counterfactual_explanation_by_paths(df, action_name=None, exclude_features=None, state_col='Game_Features'):
    explanations = []
    # Get the root node name and action space
    root_node_name = df[df['Parent_Name'] == 'None'].iloc[0].Name

    # Get children list
    children_df = df[df['Parent_Name'] == root_node_name]
    children_list = children_df['Action_Name'].values
    if len(children_list) == 1:
        return []

    explanations.append(f"There are {len(children_list)} actions available for this state.")

    # Check the value of all possible action, if the value is the same, it is just a random decision
    if len(df[df.Parent_Name == root_node_name].Value.unique()) == 1:
        explanations.append("The value of every action is the same.")

    # Get Best action name
    best_action_name = df[df['Parent_Name'] == 'None']['Best_Action'].values[0]

    # Get second action name
    if action_name:
        # If action_name is provided and it does not meet the requirement, print message and return
        if action_name not in children_list or action_name == best_action_name:
            if action_name not in children_list:
                return [f"The selected action {action_name} is not available for this state."]
            if action_name == best_action_name:
                return [f"The selected action {action_name} is not available for this state."]
            return []
    else:
        # Get the second-highest action name if action_name is not provided
        if not action_name:
            orders = children_df['Value'].argsort().values[::-1]
            action_name = children_df.iloc[orders[1]].Action_Name
            # Exception case (all value is the same and current selection is the best action)
            if action_name == best_action_name:
                action_name = children_df.iloc[orders[1]].Action_Name

    explanations.append(f"Counterfactual explanation based on {best_action_name} and {action_name}.")

    # Get the best value path of best action
    best_action_paths = get_action_path(df, action_name=best_action_name, exclude_features=exclude_features, state_col=state_col)

    # Get the best value path of selected action (or second-highest value action)
    comparison_action_paths = get_action_path(df, action_name=action_name, exclude_features=exclude_features, state_col=state_col)

    # Get path length and shorten the path
    path_len = min(len(best_action_paths), len(comparison_action_paths))
    best_action_paths = best_action_paths[:path_len]
    comparison_action_paths = comparison_action_paths[:path_len]

    ba_feature_differences, ba_percentage_differences = get_path_feature_difference(best_action_paths)
    ca_feature_differences, ca_percentage_differences = get_path_feature_difference(comparison_action_paths)

    # Get the number of executed actions in the path
    explanations.append(f"The explanation is based on the result of executing {path_len - 1} action(s) by the agent.")

    # Generate the explanation
    # Situation 1: No difference between two paths
    if ba_feature_differences == ca_feature_differences:
        feature_explanation = generate_explanation_list(ba_feature_differences, ba_percentage_differences)
        if len(feature_explanation) == 0:
            explanations.append(f"After executing {best_action_name} or {action_name}, nothing happens.")
            return explanations
        if len(feature_explanation) == 1:
            explanations.append(
                f"After executing {best_action_name} or {action_name}, I expected {feature_explanation[0]} in the future state.")
            return explanations
        explanations.append(f"After executing {best_action_name} or {action_name}, I expected {', '.join(feature_explanation[:-1])} and {feature_explanation[-1]} in the future state.")
        return explanations

    # Situation 2: Difference between two paths
    # Classify the features
    common_features, ba_exclude_features, ca_exclude_features = classify_features(ba_feature_differences, ca_feature_differences)

    if len(common_features):
        feature_explanation = generate_explanation_list(ba_feature_differences, ba_percentage_differences,
                                                        common_features, ca_percentage_differences)
        union_sentence = f"{', '.join(feature_explanation[:-1])} and {feature_explanation[-1]}" if len(
            feature_explanation) > 1 else feature_explanation[0]
        explanations.append(f"After executing {best_action_name} or {action_name}, I expect {union_sentence} in the future state.")

    if len(ba_exclude_features):
        best_feature_explanation = generate_explanation_list(ba_feature_differences, ba_percentage_differences,
                                                             ba_exclude_features)
        best_sentence = f"{', '.join(best_feature_explanation[:-1])} and {best_feature_explanation[-1]}" if len(
            best_feature_explanation) > 1 else best_feature_explanation[0]
        explanations.append(f"By executing {best_action_name}, I also expect {best_sentence}.")

    if len(ca_exclude_features):
        comparison_feature_explanation = generate_explanation_list(ca_feature_differences, ca_percentage_differences,
                                                                    ca_exclude_features)
        comparison_sentence = f"{', '.join(comparison_feature_explanation[:-1])} and {comparison_feature_explanation[-1]}" if len(
            comparison_feature_explanation) > 1 else comparison_feature_explanation[0]
        explanations.append(f"By executing {action_name}, I will expect {comparison_sentence}.")
    return explanations

