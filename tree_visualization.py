import plotly
import similarity
import pandas as pd
import numpy as np
import networkx as nx
import json
import plotly.graph_objects as go
import plotly.express as px

BASIC_ATTRIBUTE = ["Name", "Parent_Name", "Depth", "Value", "Visits"]


def get_attributes(df):
    return sorted(df.columns.tolist())


def get_root_actions(df, exclude_best=True):
    root_node = df[df['Parent_Name'] == 'None'].iloc[0]
    children = list(df[df['Parent_Name'] == root_node['Name']]['Action_Name'].values)
    if exclude_best:
        children.remove(root_node['Best_Action'])
    return children


def check_validity(filename):
    df = pd.read_csv(filename, sep="\t")
    for attribute in BASIC_ATTRIBUTE:
        if attribute not in df.columns:
            return False
    return True


def get_figure_object(fig):
    if type(fig) == dict:
        return plotly.io.from_json(json.dumps(fig))
    return fig


def highlight_selected_node(fig, node_name):
    fig = get_figure_object(fig)
    custom_data = fig.data[1]['customdata']
    fig.data[1].update(
        marker={"opacity": [0.4 if i['Name'] != node_name else 1 for i in custom_data],
                "size": [10 if i['Name'] != node_name else 20 for i in custom_data]})
    return fig


def reset_highlight_figure(fig):
    fig = get_figure_object(fig)
    custom_data = fig.data[1]['customdata']
    fig.data[1].update(
        marker={"opacity": [1 for _ in custom_data], "size": [10 for _ in custom_data]})
    return fig


def get_features(df, node_name):
    feature = df[df['Name'] == node_name]['Game_Features'].values[0]
    feature = pd.json_normalize(json.loads(feature)).T
    return feature.values.squeeze()


def search_similar_node_by_features(df, visit_threshold, node_name, similarity_method, threshold):
    if 'Game_Features' not in df.columns:
        return []

    df = df[df['Visits'] >= visit_threshold]
    feature1 = get_features(df, node_name)

    similarities = {}

    for _, row in df.iterrows():
        if row['Name'] == node_name:
            continue
        feature2 = get_features(df, row['Name'])
        value = similarity.distance_similarity_function_dict[similarity_method](feature1, feature2)
        if value <= threshold:
            similarities[row['Name']] = value

    similarities = sorted(similarities.items(), key=lambda x: x[1])
    similarities = [children[0] for children in similarities]

    return list(similarities)


def search_similar_node_by_children_action(df, visit_threshold, node_name, similarity_method, threshold):
    if 'Action_Name' not in df.columns:
        return []

    df = df[df['Visits'] >= visit_threshold]
    children_dict = {name: [] for name in df['Name']}
    action_dict = {name: idx for idx, name in enumerate(df['Action_Name'].unique())}

    for _, row in df.iterrows():
        if row['Parent_Name'] not in children_dict:
            children_dict[row['Parent_Name']] = []
        children_dict[row['Parent_Name']].append(action_dict[row['Action_Name']])

    similarities = {}

    for node, children_list in children_dict.items():
        if node == node_name:
            continue

        value = similarity.set_similarity_function_dict[similarity_method](children_dict[node_name], children_dict[node])
        if value >= threshold:
            similarities[node] = value

    similarities = sorted(similarities.items(), key=lambda x: x[1])
    similarities = [children[0] for children in similarities]

    return list(similarities)


def search_children(df, visit_threshold, node_name):
    df = df[df['Visits'] >= visit_threshold]
    return df[df['Parent_Name'] == node_name]['Name'].tolist()


def get_json_data_attribute_list(df):
    json_attribute_list = []
    attributes = get_attributes(df)
    for attribute in attributes:
        try:
            test_data = str(df[attribute].values[:1][0])
            if not test_data.startswith("{"):
                continue
            json.loads(test_data)
        except ValueError:
            continue
        json_attribute_list.append(attribute)
    return json_attribute_list


def get_numerical_only_json_data_attribute_list(df):
    consideration_list = get_json_data_attribute_list(df)
    attribute_list = []
    for attribute in consideration_list:
        data = json.loads(str(df[attribute].values[:1][0]))
        is_numerical = True
        for _, val in data.items():
            if type(val) not in [int, float]:
                is_numerical = False
                break
        if is_numerical:
            attribute_list.append(attribute)
    return attribute_list


def generate_network(df):
    dag = nx.from_pandas_edgelist(df, source='Name', target='Parent_Name')
    dag.remove_node("None")

    # Change the position of node and edge
    pos = nx.nx_agraph.graphviz_layout(dag, prog='twopi')
    for n, p in pos.items():
        dag.nodes[n]['pos'] = p

    # Put the attributes of node into network
    attributes = get_attributes(df)
    for x in df["Name"]:
        dag.nodes[x]["Name"] = x
        if attributes:
            for attribute in attributes:
                dag.nodes[x][attribute] = df[df["Name"] == x][attribute].values[0]

    return dag


def get_edge_pos(graph):
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    return edge_x, edge_y


def get_node_data(graph, df):
    node_x = []
    node_y = []
    custom_data = []
    attributes = get_attributes(df)

    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

        data = {"Name": graph.nodes[node]['Name']}
        if attributes:
            for attribute in attributes:
                data[attribute] = graph.nodes[node][attribute]
        custom_data.append(data)

    return node_x, node_y, custom_data


def generate_hover_template_text(df, hover_template_attributes):
    if hover_template_attributes:
        hover_template_text = ["%{customdata.Name}"]
        for attribute in hover_template_attributes:
            if attribute in get_attributes(df):
                hover_template_text.append(f"{attribute}: %{{customdata.{attribute}}}")
        hover_template_text = "<br>".join(hover_template_text)
        hover_template_text += "<extra></extra>"
        return hover_template_text
    return "%{customdata.Name}<extra></extra>"


def update_hover_text(fig, df, hover_template_attributes=None):
    fig = get_figure_object(fig)
    hover_template_text = generate_hover_template_text(df, hover_template_attributes)
    fig.data[1].update(hovertemplate=hover_template_text)
    return fig


def get_legend_attributes(df):
    legend_list = df.select_dtypes([np.number]).columns.tolist()
    object_type_legend_list = []
    json_attributes = get_json_data_attribute_list(df)
    for i in df.select_dtypes(exclude=[np.number]).columns:
        if i not in json_attributes and i not in ["Name", "Parent_Name"]:
            if len(df[i].unique()) < 24:
                legend_list.append(i)
                object_type_legend_list.append(i)
    return sorted(legend_list), sorted(object_type_legend_list)


def get_custom_data_by_node_name(fig, node_name):
    fig = get_figure_object(fig)
    custom_data = fig.data[1]['customdata']

    for data in custom_data:
        if data['Name'] == node_name:
            return data
    return None


def generate_fig(graph, df):
    edge_x, edge_y = get_edge_pos(graph)
    node_x, node_y, custom_data = get_node_data(graph, df)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y, customdata=custom_data,
        mode='markers',
        marker=dict(
            showscale=True,
            colorscale='bluered',
            color=[],
            size=10,
            opacity=1,
            colorbar=dict(
                thickness=15,
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    node_trace.marker.color = [i['Depth'] for i in custom_data]

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig = update_hover_text(fig, df)
    return fig


def generate_visit_threshold_network(df, threshold, hovertext=None, legend=None):
    df = df[df['Visits'] >= threshold]

    fig = generate_fig(generate_network(df), df)

    if hovertext:
        fig = update_hover_text(fig, df, hovertext)

    if legend:
        fig = update_legend(fig, df, legend)

    update_marker_symbols(fig)

    return fig


def update_legend(fig, df, legend_name, visit_threshold=None):
    fig = get_figure_object(fig)
    custom_data = fig.data[1]['customdata']

    _, object_type_legend_attributes = get_legend_attributes(df)

    if legend_name in object_type_legend_attributes:
        if visit_threshold:
            df = df[df['Visits'] >= visit_threshold]
        name_dict = {name: idx for idx, name in enumerate(df[legend_name].unique())}
        colorbar = dict(
            thickness=15,
            xanchor='left',
            titleside='right',
            tickvals=[i for i in list(name_dict.values())],
            ticktext=list(name_dict.keys())
        )
        color = [name_dict[i[legend_name]] for i in custom_data]
        colorscale = []
        for val, template_color in zip(list(name_dict.values()), px.colors.qualitative.Light24):
            colorscale.append([float(val) / len(name_dict), f"rgb{plotly.colors.hex_to_rgb(template_color)}"])
            colorscale.append([float(val + 1) / len(name_dict), f"rgb{plotly.colors.hex_to_rgb(template_color)}"])
    else:
        colorbar = dict(
            thickness=15,
            xanchor='left',
            titleside='right',
            tickvals=None,
            ticktext=None
        )
        color = [i[legend_name] for i in custom_data]
        colorscale = "bluered"

    fig.data[1].update(marker={"color": color, "colorscale": colorscale, "colorbar": colorbar})
    return fig


def update_marker_symbols(fig, markers_dict=None):
    # update root node
    fig = get_figure_object(fig)
    custom_data = fig.data[1]['customdata']
    symbols = ["circle" if i['Parent_Name'] != "None" else "circle-x" for i in custom_data]
    fig.data[1].update(marker={"symbol": symbols})
    return fig
