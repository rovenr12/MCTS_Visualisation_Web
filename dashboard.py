import base64
import os.path

import similarity
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import tree_visualization
import explanation
import plotly.graph_objects as go
import json
import pandas as pd

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
load_figure_template("BOOTSTRAP")

DISTANCE_SIMILARITY_METHODS = list(similarity.distance_similarity_function_dict.keys())
SET_SIMILARITY_METHODS = list(similarity.set_similarity_function_dict.keys())


####################################################
# Helper functions
####################################################
def create_select_options(options):
    return [{"label": str(option), "value": option} for option in options]


def json_content_builder(json_obj, container):
    for key, value in json_obj.items():
        if type(value) == list and type(value[0]) == dict:
            sub_list_group = dbc.ListGroup(children=[])
            for idx, child in enumerate(value, start=1):
                child_list_group = dbc.ListGroup(children=[])
                json_content_builder(child, child_list_group)
                sub_list_group.children.append(dbc.ListGroupItem([html.P(f"{key} {idx}", className='m-0 fw-bold'),
                                                                  child_list_group]))
            container.children.append(dbc.ListGroupItem([html.P(key, className='m-0 fw-bold'), sub_list_group]))
        else:
            if type(value) == list:
                item_list = html.Ul([])
                for v in value:
                    item_list.children.append(html.Li(html.Small(str(v))))
                container.children.append(dbc.ListGroupItem([html.P(key, className='m-0 fw-bold'), item_list]))
            else:
                container.children.append(
                    dbc.ListGroupItem([html.P(key, className='m-0 fw-bold'), html.Small(str(value))]))


####################################################
# Configuration Card Layout
####################################################
# The layout for upload file
upload_file = html.Div([
    dbc.Label("MCTS file", html_for='upload_data', class_name='mb-1'),
    dcc.Upload([
        dbc.Button("Upload File", outline=True, color="primary", className="mb-2", id='upload_data_button'),
        dbc.Popover([
            dbc.PopoverHeader("MCTS File Requirement", class_name='bg-info'),
            dbc.PopoverBody("The MCTS file needs to be a csv file and seperated by '\\t'. The file also needs to " +
                            "have following columns at least: Name, Parent_Name, Depth, Value and Visit")
        ], target='upload_data_button', trigger='hover')
    ], id='upload_data'),
    html.Div(id='output_data_upload'),
    dcc.Store("filename"),
    dcc.Store("dataframe")
], className='py-1')

# The layout for configuration
# hover text config layout
hover_text_config = html.Div([
    dbc.Label("Node Hover Text", html_for='hover_text', class_name='mb-1'),
    dcc.Dropdown(id='hover_text', options=[], multi=True)
], className='py-1')

# legend config layout
legend_config = html.Div([
    dbc.Label("Legend", html_for='legend', class_name='mb-1'),
    dbc.Select(id='legend', options=['Depth'], value='Depth'),
    dbc.Popover([
        dbc.PopoverHeader("Categorical Legend Limitation", class_name='bg-info'),
        dbc.PopoverBody("Only available to set the categorical data as a legend if it contains less than 24 unique "
                        "values")
    ], target='legend', trigger='hover')
], className='py-1')

# visit threshold layout
visit_threshold_config = html.Div([
    dbc.Label("Visit Threshold", html_for='visit_threshold', class_name='mb-1'),
    dbc.Input(id='visit_threshold', min=1, max=20, value=1, step=1, type='number'),
    dbc.FormText('Type number between 1 to 20', id='visit_threshold_form_text')
], className='py-1')

# node configuration layout
node_configuration = html.Div([
    hover_text_config,
    legend_config,
    visit_threshold_config
], id='node_config', hidden=True, className='py-1')

# configuration card layout
configuration_card = dbc.Card([
    dbc.CardHeader(html.H4("Configurations", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody([
        node_configuration,
        upload_file
    ])
], class_name='my-2 shadow h-100')

####################################################
# Tree Visualisation Layout
####################################################
tree_visualisation = html.Div(dbc.Card([
    dbc.CardHeader(html.H4("Tree Visualisation", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody([
        html.Div(dbc.Alert("Updating graph..."), hidden=True, id='graph_alert',
                 className='text-center position-absolute top-50 start-50 translate-middle w-75'),
        html.Div(dcc.Graph(id='tree_visualisation_graph', figure=go.Figure()), id='graph_holder', hidden=True),
        dcc.Store(id='figure_filename')
    ], id='output-graph')
], class_name='h-100'), id='tree_visualisation', hidden=True, className='my-2 shadow h-100')

####################################################
# Selected Node Layout
####################################################
# Node detail layout
node_detail_card = dbc.Card([
    dbc.CardHeader(html.H6("Detail", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody(id='node_information', style={'overflow-y': 'auto', 'height': '770px'})
], class_name='my-2')

# children list layout
children_nodes = dbc.Card([
    dbc.CardHeader(html.H6("Children", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody(id='children_buttons', style={'overflow-y': 'auto', 'height': '200px'}),
    dcc.Store(id='children_button_names')
], class_name='my-2')

# similar node - game feature
game_feature_similarity_function = dbc.Select(
    id='game_feature_similarity_function',
    options=create_select_options(DISTANCE_SIMILARITY_METHODS),
    value=DISTANCE_SIMILARITY_METHODS[0]
)
game_feature_similarity_textbox = dbc.Input(type='number', id='game_feature_similarity_threshold', value=0, min=0)
similar_nodes_game_features_accordion = html.Div([
    dbc.Accordion([
        dbc.AccordionItem([
            "Similarity Function", game_feature_similarity_function,
            "Similarity Threshold", game_feature_similarity_textbox
        ], id='similar_game_features_configuration', title='Configuration'),
        dbc.AccordionItem(id='similar_game_features_content', title='Nodes'),
    ], always_open=True, active_item=['item-0'], id='similar_game_features_accordion'),
    dcc.Store(id='game_features_button_names')
], id="similar_nodes_game_features_accordion", hidden=True)

similar_nodes_game_features_warning = html.Div(dbc.Alert("Not Available. Required Game_Features column. The "
                                                         "Game_Features must be json format text, which key is name "
                                                         "of feature and value is numerical value", color="info"),
                                               id="similar_nodes_game_features_warning")

similar_nodes_game_features_card = dbc.Card([
    dbc.CardHeader(html.P("Game Features", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody([similar_nodes_game_features_accordion, similar_nodes_game_features_warning])
], class_name='my-2')

#  similar node - children action
children_action_similarity_function = dbc.Select(
    id='children_action_similarity_function',
    options=create_select_options(SET_SIMILARITY_METHODS),
    value=SET_SIMILARITY_METHODS[0]
)
children_action_similarity_textbox = dbc.Input(type='number', id='children_action_similarity_threshold',
                                               value=1, min=0, max=1)
similar_nodes_children_action_accordion = html.Div([
    dbc.Accordion([
        dbc.AccordionItem([
            "Similarity Function", children_action_similarity_function,
            "Similarity Threshold", children_action_similarity_textbox
        ], id='similar_children_action_configuration', title='Configuration'),
        dbc.AccordionItem(id='similar_children_action_content', title='Nodes'),
    ], always_open=True, active_item=['item-0'], id='similar_children_action_accordion'),
    dcc.Store(id='children_action_button_names')
], id='similar_nodes_children_action_accordion', hidden=False)

similar_nodes_children_action_warning = html.Div(dbc.Alert("Not Available. Required Action_Name column.", color="info"),
                                                 id="similar_nodes_children_action_warning")

similar_nodes_children_action_card = dbc.Card([
    dbc.CardHeader(html.P("Children Actions", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody([similar_nodes_children_action_accordion, similar_nodes_children_action_warning]),
], class_name='my-3')

# similar node card
similar_nodes = dbc.Card([
    dbc.CardHeader(html.H6("Similar Nodes", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody([similar_nodes_game_features_card, similar_nodes_children_action_card],
                 style={'overflow-y': 'auto', 'height': '500px'})
], class_name='mt-3')

# select node card body layout
select_node_info = html.Div([
    dbc.Row([
        dbc.Col(node_detail_card, lg='6', md='12'),
        dbc.Col(html.Div([children_nodes, similar_nodes]), lg='6', md='12'),
    ], class_name='g-3')
], className='p-0 m-0', id='select_node_info', hidden=True,
    style={'overflow-x': 'hidden', 'overflow-y': 'auto'})

# Select node card layout
selected_node_info_card = html.Div(dbc.Card([
    dbc.CardHeader([
        html.H4(["Selected Node Information", html.Span(hidden=True, id='select_node_badge')],
                className='m-0 fw-bold text-center text-primary'),
    ], class_name='py-3'),
    dbc.CardBody(select_node_info),
    dcc.Store(id='selected_node_custom_data')
]), id='selected_node_info_card', hidden=True, className='my-2 shadow')

####################################################
# Explanation Layout
####################################################
# Game Feature Explanation
game_feature_explanation = dbc.Card([
    dbc.CardBody(id='game_feature_explanation')
], class_name='my-2')

# feature column config
path_feature_col_config = html.Div([
    dbc.Label("Features Column Name", html_for='path_feature_column', class_name='mb-1'),
    dbc.Select(id='path_feature_column', options=[])
], className='py-1')

# type config
path_type_config = html.Div([
    dbc.Label("Type", html_for='path_type', class_name='mb-1'),
    dbc.Select(id='path_type',
               options=['Best path vs Worse path', 'Best action vs Second best action', 'Best action vs another action'],
               value='Best path vs Worse path')
], className='py-1')

# action name config
path_action_name_config = html.Div([
    dbc.Label("Compare Action Name", html_for='path_action_name', class_name='mb-1'),
    dbc.Select(id='path_action_name', options=[])
], className='py-1', hidden=True, id='path_action_name_div')

# exclude features config
exclude_features_config = html.Div([
    dbc.Label("Exclude Features", html_for='path_exclude_features', class_name='mb-1'),
    dcc.Dropdown(id='path_exclude_features', options=[], multi=True)
], className='py-1')

# generation button
path_explanation_generation = html.Div([
    dbc.Button("Generate", color="primary", id='path_explanation_generation')
], className='d-grid pt-4 pb-3')

path_configuration_card = dbc.Card([
    dbc.CardHeader(html.P("Configuration", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody([
        path_feature_col_config,
        path_type_config,
        path_action_name_config,
        exclude_features_config,
        path_explanation_generation
    ], id='path_configuration_card')
], class_name='my-2')

path_explanation_card = dbc.Card([
    dbc.CardHeader(html.P("Explanation", className='m-0 fw-bold text-center text-primary'), class_name='py-3'),
    dbc.CardBody(id='path_explanation_card')
], class_name='my-2')

path_explanation = dbc.Card([
    dbc.CardBody(dbc.Container([
        dbc.Row([
            dbc.Col(path_configuration_card, lg='4', md='12'),
            dbc.Col(path_explanation_card, lg='8', md='12')
        ], class_name='g-3 mb-4 px-3')
    ]), id='path_explanation')
], class_name='my-2')

explanation_tabs = dbc.Tabs([
    dbc.Tab(game_feature_explanation, label='Game Features'),
    dbc.Tab(path_explanation, label='Path')
])

explanation_card = html.Div(dbc.Card([
    dbc.CardHeader(html.H4("Root Action Explanation", className='m-0 fw-bold text-center text-primary'),
                   class_name='py-3'),
    dbc.CardBody([
        explanation_tabs
    ], style={'height': '700px', 'overflow-x': 'hidden', 'overflow-y': 'auto'})
], className='h-100'), id='root_action_explanation', hidden=True, className='my-2 shadow h-100')

####################################################
# App Layout
####################################################
app.layout = html.Div([
    dbc.Row(dbc.Col(html.H1('MCTS Dashboard')), class_name='text-center py-2 bg-primary mb-3'),
    dbc.Container([
        dbc.Row([
            dbc.Col(configuration_card, lg='3', md='12'),
            dbc.Col(tree_visualisation, lg='9', md='12')
        ], class_name='g-3 mb-4 px-3'),
        dbc.Row([
            dbc.Col(selected_node_info_card, lg='12', md='12'),
        ], class_name='g-3 mb-4 px-3'),
        dbc.Row([
            dbc.Col(explanation_card, lg='12', md='12')
        ], class_name='g-3 mb-4 px-3')
    ])
], style={"overflow-x": "hidden", "min-height": "100vh"}, className="bg-light")


####################################################
# Explanation Callback
####################################################
@app.callback(
    Output(component_id='path_action_name_div', component_property='hidden'),
    Input(component_id='path_type', component_property='value')
)
def hidden_action_name_config(path_type):
    return path_type != 'Best action vs another action'

@app.callback(
    Output(component_id='game_feature_explanation', component_property='children'),
    Input(component_id='root_action_explanation', component_property='hidden'),
    Input(component_id='visit_threshold', component_property='value'),
    State(component_id='dataframe', component_property='data')
)
def update_game_feature_explanation(hidden, visit_threshold, df):
    if hidden:
        return []

    df = pd.read_json(df)

    if 'Game_Features' not in df.columns:
        return dbc.Alert("Not Available. Required Game_Features column.", color="info")

    if visit_threshold:
        df = df[df['Visits'] >= visit_threshold]

    root_action_list = explanation.get_root_children_list(df)
    if root_action_list:
        table_df, maximum_depth = explanation.generate_game_feature_explanation_df(df)
        maximum_depth += 1
        table = dbc.Table.from_dataframe(table_df, bordered=True, hover=True, index=True,
                                         responsive=True, class_name='align-middle text-center')
        tbody = table.children[1]
        tbody.className = " table-group-divider"

        visited_feature_name = []

        for idx, tr in enumerate(tbody.children):
            if tr.children[0].children not in visited_feature_name:
                visited_feature_name.append(tr.children[0].children)
                tr.children[0].rowSpan = maximum_depth
            else:
                tr.children.pop(0)
        return table

    return []


####################################################
# Node Detail Callback
####################################################
@app.callback(
    Output(component_id='node_information', component_property='children'),
    Input(component_id='select_node_info', component_property='hidden'),
    State(component_id='dataframe', component_property='data'),
    State(component_id='selected_node_custom_data', component_property='data')
)
def update_node_information(hidden, df, selected_node):
    if hidden:
        return ""

    accordion = dbc.Accordion([], always_open=True, active_item=[])
    df = pd.read_json(df)
    attributes = tree_visualization.get_attributes(df)

    for attribute in attributes:
        if attribute == 'Name':
            continue
        elif attribute in tree_visualization.get_json_data_attribute_list(df):
            game_state_json = json.loads(selected_node[attribute])
            list_group = dbc.ListGroup(children=[])
            json_content_builder(game_state_json, list_group)
            accordion.children.append(dbc.AccordionItem(list_group, title=attribute))
        else:
            accordion.children.append(dbc.AccordionItem(selected_node[attribute], title=attribute))

    return accordion


####################################################
# Similar Node (Game Features) Callback
####################################################
#  layout
@app.callback(
    Output(component_id='similar_nodes_game_features_accordion', component_property='hidden'),
    Output(component_id='similar_nodes_game_features_warning', component_property='hidden'),
    Input(component_id='select_node_info', component_property='hidden'),
    State(component_id='dataframe', component_property='data')
)
def update_node_similar_game_feature_layout(hidden, df):
    if hidden:
        return True, True

    df = pd.read_json(df)

    if 'Game_Features' not in df.columns:
        return True, False

    if 'Game_Features' not in tree_visualization.get_json_data_attribute_list(df):
        return True, False

    return False, True


# configuration value reset
@app.callback(
    Output(component_id='game_feature_similarity_function', component_property='value'),
    Output(component_id='game_feature_similarity_threshold', component_property='value'),
    Output(component_id='similar_game_features_accordion', component_property='active_item'),
    Input(component_id='upload_data', component_property='filename')
)
def update_graph_alert(_):
    return DISTANCE_SIMILARITY_METHODS[0], 0, ['item-0']


# content
@app.callback(
    Output(component_id='similar_game_features_content', component_property='children'),
    Output(component_id='game_features_button_names', component_property='data'),
    Input(component_id='similar_nodes_game_features_accordion', component_property='hidden'),
    Input(component_id='game_feature_similarity_function', component_property='value'),
    Input(component_id='game_feature_similarity_threshold', component_property='value'),
    State(component_id='dataframe', component_property='data'),
    State(component_id='visit_threshold', component_property='value'),
    State(component_id='selected_node_custom_data', component_property='data')
)
def update_node_similar_game_feature_content(hidden, similarity_name, similarity_threshold,
                                             df, visit_threshold, selected_node):
    if hidden:
        return "", []

    df = pd.read_json(df)
    if 'Game_Features' not in df.columns:
        return "", []

    if not similarity_name or not similarity_threshold:
        return "", []

    if not visit_threshold:
        visit_threshold = 1

    similar_nodes_by_features = tree_visualization.search_similar_node_by_features(df, visit_threshold,
                                                                                   selected_node['Name'],
                                                                                   similarity_name,
                                                                                   similarity_threshold)

    similar_game_features_button = []
    similar_game_features_button_names = []
    for idx, children in enumerate(similar_nodes_by_features):
        similar_game_features_button.append(dbc.Col(dbc.Button(children, size="sm",
                                                               id={'type': 'game_feature_button', 'index': idx}),
                                                    className="m-2"))
        similar_game_features_button_names.append(children)

    return dbc.Row(similar_game_features_button, class_name='row-cols-auto g-1',
                   justify='center'), similar_game_features_button_names


####################################################
# Similar Node (Children Action) Callback
####################################################
#  layout
@app.callback(
    Output(component_id='similar_nodes_children_action_accordion', component_property='hidden'),
    Output(component_id='similar_nodes_children_action_warning', component_property='hidden'),
    Input(component_id='select_node_info', component_property='hidden'),
    State(component_id='dataframe', component_property='data')
)
def update_node_similar_game_feature_layout(hidden, df):
    if hidden:
        return True, True

    df = pd.read_json(df)

    if 'Action_Name' not in df.columns:
        return True, False

    return False, True


# configuration value reset
@app.callback(
    Output(component_id='children_action_similarity_function', component_property='value'),
    Output(component_id='children_action_similarity_threshold', component_property='value'),
    Output(component_id='similar_children_action_accordion', component_property='active_item'),
    Input(component_id='upload_data', component_property='filename')
)
def update_graph_alert(_):
    return SET_SIMILARITY_METHODS[0], 1, ['item-0']


# content
@app.callback(
    Output(component_id='similar_children_action_content', component_property='children'),
    Output(component_id='children_action_button_names', component_property='data'),
    Input(component_id='similar_nodes_children_action_accordion', component_property='hidden'),
    Input(component_id='children_action_similarity_function', component_property='value'),
    Input(component_id='children_action_similarity_threshold', component_property='value'),
    State(component_id='dataframe', component_property='data'),
    State(component_id='visit_threshold', component_property='value'),
    State(component_id='selected_node_custom_data', component_property='data')
)
def update_node_similar_game_feature_content(hidden, similarity_name, similarity_threshold,
                                             df, visit_threshold, selected_node):
    if hidden:
        return "", []

    df = pd.read_json(df)
    if 'Action_Name' not in df.columns:
        return "", []

    if not similarity_name or not similarity_threshold:
        return "", []

    if not visit_threshold:
        visit_threshold = 1

    similar_nodes_by_children_action = tree_visualization.search_similar_node_by_children_action(df, visit_threshold,
                                                                                                 selected_node['Name'],
                                                                                                 similarity_name,
                                                                                                 similarity_threshold)

    similar_children_action_button = []
    similar_children_action_button_names = []
    for idx, children in enumerate(similar_nodes_by_children_action):
        similar_children_action_button.append(dbc.Col(dbc.Button(children, size="sm",
                                                                 id={'type': 'children_action_button', 'index': idx}),
                                                      className="m-2"))
        similar_children_action_button_names.append(children)

    return dbc.Row(similar_children_action_button, class_name='row-cols-auto g-1',
                   justify='center'), similar_children_action_button_names


####################################################
# Children Node Callback
####################################################
@app.callback(
    Output(component_id='children_buttons', component_property='children'),
    Output(component_id='children_button_names', component_property='data'),
    Input(component_id='select_node_info', component_property='hidden'),
    State(component_id='dataframe', component_property='data'),
    State(component_id='visit_threshold', component_property='value'),
    State(component_id='selected_node_custom_data', component_property='data')
)
def update_node_children(hidden, df, visit_threshold, selected_node):
    if hidden:
        return "", []

    children_buttons = []
    children_button_names = []
    df = pd.read_json(df)

    if not visit_threshold:
        visit_threshold = 1

    for idx, children in enumerate(tree_visualization.search_children(df, visit_threshold, selected_node['Name'])):
        children_buttons.append(dbc.Col(dbc.Button(children, size="sm",
                                                   id={'type': 'children_button', 'index': idx}), className="m-2"))
        children_button_names.append(children)

    return dbc.Row(children_buttons, class_name='row-cols-auto g-1', justify='center'), children_button_names


####################################################
# Selected Node Callback
####################################################
@app.callback(
    Output(component_id='select_node_badge', component_property='children'),
    Output(component_id='select_node_badge', component_property='hidden'),
    Output(component_id='select_node_info', component_property='hidden'),
    Output(component_id='selected_node_custom_data', component_property='data'),
    Input(component_id='visit_threshold', component_property='value'),
    Input(component_id='tree_visualisation_graph', component_property='clickData'),
    Input(component_id={'type': 'children_button', 'index': ALL}, component_property='n_clicks'),
    Input(component_id={'type': 'game_feature_button', 'index': ALL}, component_property='n_clicks'),
    Input(component_id={'type': 'children_action_button', 'index': ALL}, component_property='n_clicks'),
    State(component_id='children_button_names', component_property='data'),
    State(component_id='game_features_button_names', component_property='data'),
    State(component_id='children_action_button_names', component_property='data'),
    State(component_id='tree_visualisation_graph', component_property='figure'),
    State(component_id='selected_node_custom_data', component_property='data'),
    State(component_id='select_node_badge', component_property='children')
)
def update_select_node_budget_status(_, click_data, children_button, game_feature_button, children_action_button,
                                     children_button_names, game_feature_button_names, children_action_button_names,
                                     fig, selected_node, current_badge):
    if not ctx.triggered_id:
        return current_badge, False, False, selected_node

    if ctx.triggered_id == 'visit_threshold':
        return "", True, True, None

    if ctx.triggered_id == "tree_visualisation_graph":
        if not click_data:
            return "", True, True, None

        if 'customdata' not in click_data['points'][0]:
            return "", True, True, None

        selected_node = click_data['points'][0]['customdata']
    elif ctx.triggered_id['type'] == 'children_button' and 1 in children_button:
        name = children_button_names[ctx.triggered_id['index']]
        selected_node = tree_visualization.get_custom_data_by_node_name(fig, name)
    elif ctx.triggered_id['type'] == 'game_feature_button' and 1 in game_feature_button:
        name = game_feature_button_names[ctx.triggered_id['index']]
        selected_node = tree_visualization.get_custom_data_by_node_name(fig, name)
    elif ctx.triggered_id['type'] == 'children_action_button' and 1 in children_action_button:
        name = children_action_button_names[ctx.triggered_id['index']]
        selected_node = tree_visualization.get_custom_data_by_node_name(fig, name)

    # Generate Budge
    budge = html.Span(dbc.Badge(selected_node['Name'], color='primary', className="ms-2", pill=True))

    return budge, False, False, selected_node


####################################################
# Tree visualisation Callback
####################################################
# showing alert during graph generation after loading file
@app.callback(
    Output(component_id='graph_alert', component_property='hidden'),
    Output(component_id='graph_holder', component_property='hidden'),
    Input(component_id='upload_data', component_property='filename'),
    Input(component_id='tree_visualisation_graph', component_property='figure')
)
def update_graph_alert(_, figure):
    if ctx.triggered_id == 'upload_data':
        return False, True

    if not figure or not figure['data']:
        return True, True

    return True, False


# graph generation
@app.callback(
    Output(component_id='tree_visualisation_graph', component_property='figure'),
    Output(component_id='figure_filename', component_property='data'),
    Input(component_id='hover_text', component_property='value'),
    Input(component_id='legend', component_property='value'),
    Input(component_id='visit_threshold', component_property='value'),
    Input(component_id='select_node_info', component_property='hidden'),
    State(component_id='tree_visualisation_graph', component_property='figure'),
    State(component_id='filename', component_property='data'),
    State(component_id='figure_filename', component_property='data'),
    State(component_id='dataframe', component_property='data'),
    State(component_id='selected_node_custom_data', component_property='data')
)
def update_tree_visualization(hover_text, legend, visit_threshold, _, fig,
                              filename, figure_filename, df, selected_node):
    if not filename:
        return go.Figure(), None

    df = pd.read_json(df)
    if ctx.triggered_id == 'visit_threshold' or filename != figure_filename:
        if not visit_threshold:
            visit_threshold = 1
        return tree_visualization.generate_visit_threshold_network(df, visit_threshold, hover_text, legend), filename

    if ctx.triggered_id == "select_node_info" and selected_node:
        return tree_visualization.highlight_selected_node(fig, selected_node['Name']), figure_filename

    if ctx.triggered_id == "hover_text":
        return tree_visualization.update_hover_text(fig, df, hover_text), figure_filename

    if ctx.triggered_id == "legend":
        return tree_visualization.update_legend(fig, df, legend, visit_threshold), figure_filename

    return fig, figure_filename


####################################################
# Configuration Callback
####################################################
# Update visit_threshold marks
@app.callback(
    Output(component_id='visit_threshold_form_text', component_property='children'),
    Input(component_id='visit_threshold', component_property='max')
)
def create_visit_threshold_placeholder(max_val):
    return f'Type number between 1 and {max_val}'


# Update configuration
@app.callback(
    Output(component_id='hover_text', component_property='options'),
    Output(component_id='hover_text', component_property='value'),
    Output(component_id='legend', component_property='options'),
    Output(component_id='legend', component_property='value'),
    Output(component_id='visit_threshold', component_property='max'),
    Output(component_id='visit_threshold', component_property='value'),
    Output(component_id='path_feature_column', component_property='options'),
    Output(component_id='path_feature_column', component_property='value'),
    Input(component_id='node_config', component_property='hidden'),
    State(component_id='dataframe', component_property='data')
)
def update_config_options(is_hidden, df):
    if is_hidden or not df:
        return [], [], [], 'Depth', 1, 1, [], None

    # Update Attributes
    df = pd.read_json(df)
    hover_options = []

    for attribute in tree_visualization.get_attributes(df):
        if 'Name' == attribute:
            continue
        hover_options.append({'label': attribute, 'value': attribute})

    legend_attributes, _ = tree_visualization.get_legend_attributes(df)
    legend_options = create_select_options(legend_attributes)

    path_attributes = tree_visualization.get_json_data_attribute_list(df)
    path_options = create_select_options(path_attributes)
    path_value = path_attributes[0]

    if 'Game_Features' in path_attributes:
        path_value = 'Game_Features'
    elif 'Game_State' in path_attributes:
        path_value = 'Game_State'

    visit_maximum = df['Visits'].max()
    return hover_options, [], legend_options, 'Depth', visit_maximum, 1, path_options, path_value


####################################################
# Upload file Callback
####################################################
@app.callback(
    Output(component_id='output_data_upload', component_property='children'),
    Output(component_id='node_config', component_property='hidden'),
    Output(component_id='selected_node_info_card', component_property='hidden'),
    Output(component_id='tree_visualisation', component_property='hidden'),
    Output(component_id='root_action_explanation', component_property='hidden'),
    Output(component_id='tree_visualisation_graph', component_property='clickData'),
    Output(component_id='filename', component_property='data'),
    Output(component_id='dataframe', component_property='data'),
    Input('upload_data', 'contents'),
    State('upload_data', 'filename')
)
def upload_and_save_file(content, file_name):
    if content:
        # Save the file
        data = content.encode("utf8").split(b";base64,")[1]
        filename = f"save_{file_name}"
        with open(os.path.join(filename), 'wb') as fp:
            fp.write(base64.decodebytes(data))

        # Check Data Validity
        if not tree_visualization.check_validity(filename):
            os.remove(filename)
            alert = dbc.Alert(
                f"Invalid Data",
                color="danger", dismissable=True)
            return alert, True, True, True, True, None, None, None

        df = pd.read_csv(filename, sep="\t")
        alert = dbc.Alert(f"{file_name} has uploaded successfully!!", color="success", dismissable=True)

        return alert, False, False, False, False, None, filename, df.to_json()

    return "", True, True, True, True, None, None, None


if __name__ == '__main__':
    app.run_server(debug=True)
