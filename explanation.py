import math
import pandas as pd
import json


def get_children_in_depth(df, depth, key, children_in_depth_dict):
    if not children_in_depth_dict[key][depth]:
        return depth - 1

    children_from_current_depth_list = []
    for children in children_in_depth_dict[key][depth]:
        children_from_current_depth_list += list(df[df['Parent_Name'] == children]['Name'].values)

    children_in_depth_dict[key][depth + 1] = children_from_current_depth_list
    return get_children_in_depth(df, depth + 1, key, children_in_depth_dict)


def get_feature_difference_count(df, base_action_name, action_lists, rel_tol=0.001):
    base_game_feature = df[df['Name'] == base_action_name]['Game_Features'].values[0]
    base_game_feature = pd.json_normalize(json.loads(base_game_feature))

    game_feature_summary = {}
    for col_name in base_game_feature.columns:
        game_feature_summary[col_name] = {"higher": 0, "same": 0, "lower": 0}

    for action in action_lists:
        compare_game_feature = df[df['Name'] == action]['Game_Features'].values[0]
        compare_game_feature = pd.json_normalize(json.loads(compare_game_feature))

        for col_name in base_game_feature.columns:
            base_value = base_game_feature[col_name].values[0]
            compare_value = compare_game_feature[col_name].values[0]
            if math.isclose(base_value, compare_value, rel_tol=rel_tol):
                game_feature_summary[col_name]['same'] += 1
            elif compare_value > base_value:
                game_feature_summary[col_name]['higher'] += 1
            else:
                game_feature_summary[col_name]['lower'] += 1

    return game_feature_summary


def game_feature_difference_explanation(df):

    root_name = df[df['Parent_Name'] == 'None']['Name'][0]
    root_children = df[df['Parent_Name'] == root_name]['Name'].values

    children_in_depth_dict = {name: {0: [name]} for name in root_children}
    immediate_difference = {}
    game_feature_difference_in_depth_dict = {name: {} for name in root_children}

    for child in root_children:
        children_depth = get_children_in_depth(df, 0, child, children_in_depth_dict)

        immediate_difference[child] = get_feature_difference_count(df, root_name, [child])

        for depth in range(1, children_depth + 1):
            counts = get_feature_difference_count(df, root_name, children_in_depth_dict[child][depth])
            game_feature_difference_in_depth_dict[child][depth] = counts

    return immediate_difference, game_feature_difference_in_depth_dict


def generate_immediate_dataframe(immediate_difference):
    immediate_df = {"Name": [], "Change": [], "Feature_List": [], "Count": []}

    for node, summary in immediate_difference.items():
        name = [node] * 3
        feature_list = [[] for _ in range(3)]
        for feature, change_dict in summary.items():
            if change_dict['higher']:
                feature_list[0].append(feature)
            elif change_dict['same']:
                feature_list[1].append(feature)
            else:
                feature_list[2].append(feature)

        immediate_df['Name'] += name
        immediate_df['Change'] += ['Higher', 'Same', 'Lower']
        immediate_df['Feature_List'] += feature_list
        immediate_df['Count'] += [len(feature) for feature in feature_list]

    return pd.DataFrame.from_dict(immediate_df)


def get_root_children_list(df):
    root_name = df[df['Parent_Name'] == 'None']['Name'][0]
    return list(df[df['Parent_Name'] == root_name]['Name'].values)


if __name__ == '__main__':
    df = pd.read_csv("MCTS_test_0.csv", sep='\t')
    immediate, depth = game_feature_difference_explanation(df)
    print(generate_immediate_dataframe(immediate))
