# --------------------
# IMPORTS
# --------------------
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# --------------------
# SHARED HELPER FUNCTION (DEFINED GLOBALLY)
# --------------------
def calculate_distance_in_miles(lat1, lon1, lat2, lon2):
    """Calculates the approximate distance in miles between two coordinates."""
    lat_scale = 69 
    lon_scale = 55
    return np.sqrt(((lat1 - lat2) * lat_scale)**2 + ((lon1 - lon2) * lon_scale)**2)

# --------------------
# 1. DATA PREPARATION FUNCTION
# --------------------
def prepare_data(num_orders=200):
    try:
        df_all_orders = pd.read_csv('orders_for_app.csv')
    except FileNotFoundError:
        return None, None, None, "Error: 'orders_for_app.csv' not found. Please run the notebook pre-processing step."

    warehouse_data = {'Warehouse_ID': ['DEPOT_MANHATTAN', 'DEPOT_QUEENS', 'DEPOT_BROOKLYN', 'DEPOT_BRONX', 'DEPOT_STATEN_ISLAND'],
                      'Latitude': [40.7831, 40.7282, 40.6782, 40.8448, 40.5795], 
                      'Longitude': [-73.9712, -73.7949, -73.9442, -73.8648, -74.1502]}
    warehouses_df = pd.DataFrame(warehouse_data)
    
    orders_df = df_all_orders.sample(n=num_orders, random_state=42)
    np.random.seed(42)
    orders_df['Total_Demand'] = np.random.randint(50, 100, size=len(orders_df))
    orders_df.reset_index(drop=True, inplace=True)

    all_locations_df = pd.concat([
        warehouses_df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}),
        orders_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    ], ignore_index=True)

    return orders_df, warehouses_df, all_locations_df, None

# --------------------
# 2. SOLVER FUNCTIONS
# --------------------
def solve_with_or_tools(data, all_locations_df, orders_df, warehouses_df):
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node] * 1000)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(25)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution: return None, 0

    routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        if len(route) > 2: routes.append(route)
    
    fig = px.scatter_mapbox(orders_df, lat="latitude", lon="longitude", hover_name="Zone",
                          color_discrete_sequence=["#007BFF"], zoom=9, height=600)
    fig.add_trace(px.scatter_mapbox(warehouses_df, lat="Latitude", lon="Longitude", hover_name="Warehouse_ID",
                                  color_discrete_sequence=["#FF0000"], size_max=20).data[0])
    color_scale = px.colors.qualitative.Vivid
    for i, route in enumerate(routes):
        route_df = all_locations_df.iloc[route]
        fig.add_trace(go.Scattermapbox(mode="lines", lon=route_df['lon'], lat=route_df['lat'],
                                     line=dict(width=2, color=color_scale[i % len(color_scale)]), name=f"Vehicle {i+1}"))
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0},
                      title=f"Optimized Routes | Total Distance: {solution.ObjectiveValue() / 1000:.2f} miles")
    return fig, solution.ObjectiveValue() / 1000

def solve_with_greedy(data, all_locations_df, orders_df, warehouses_df):
    unvisited_orders_indices = list(range(len(warehouses_df), len(all_locations_df)))
    routes = []
    total_distance = 0

    for _ in range(data['num_vehicles']):
        if not unvisited_orders_indices: break
        route, current_load, current_loc_idx = [0], 0, 0
        while True:
            nearest_order_idx, min_dist = -1, float('inf')
            for order_idx in unvisited_orders_indices:
                demand = data['demands'][order_idx]
                if current_load + demand <= data['vehicle_capacities'][0]:
                    dist = data['distance_matrix'][current_loc_idx][order_idx]
                    if dist < min_dist:
                        min_dist, nearest_order_idx = dist, order_idx
            if nearest_order_idx != -1:
                route.append(nearest_order_idx)
                current_load += data['demands'][nearest_order_idx]
                total_distance += min_dist
                current_loc_idx = nearest_order_idx
                unvisited_orders_indices.remove(nearest_order_idx)
            else: break
        route.append(0)
        total_distance += data['distance_matrix'][current_loc_idx][0]
        if len(route) > 2: routes.append(route)
    
    fig = px.scatter_mapbox(orders_df, lat="latitude", lon="longitude", hover_name="Zone", color_discrete_sequence=["#007BFF"], zoom=9, height=600)
    fig.add_trace(px.scatter_mapbox(warehouses_df, lat="Latitude", lon="Longitude", hover_name="Warehouse_ID", color_discrete_sequence=["#FF0000"], size_max=20).data[0])
    color_scale = px.colors.qualitative.Vivid
    for i, route in enumerate(routes):
        route_df = all_locations_df.iloc[route]
        fig.add_trace(go.Scattermapbox(mode="lines", lon=route_df['lon'], lat=route_df['lat'],
                                     line=dict(width=2, color=color_scale[i % len(color_scale)]), name=f"Vehicle {i+1}"))
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0},
                      title=f"Non-Optimized (Greedy) Routes | Total Distance: {total_distance:.2f} miles")
    return fig, total_distance

# --------------------
# APP SETUP & LAYOUT
# --------------------
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
app.layout = html.Div([
    html.H1('Medical Supply Chain Optimizer', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Dropdown(id='solver-strategy-dropdown', options=[{'label': 'Optimized (OR-Tools)', 'value': 'optimized'},
                                                                {'label': 'Non-Optimized (Greedy)', 'value': 'greedy'}], value='optimized'),
            html.Br(), html.Label('Number of Vehicles:'),
            dcc.Input(id='num-vehicles-input', type='number', value=25, style={'width': '100%'}),
            html.Br(), html.Br(), html.Label('Vehicle Capacity (units):'),
            dcc.Input(id='vehicle-capacity-input', type='number', value=800, style={'width': '100%'}),
            html.Br(), html.Br(),
            html.Button('Run Solver', id='run-button', n_clicks=0, style={'width': '100%'})
        ], className='four columns'),
        html.Div([dcc.Loading(dcc.Graph(id='output-map'))], className='eight columns')
    ], className='row', style={'padding': '20px'}),
    html.Div(id='summary-stats', style={'textAlign': 'center', 'fontWeight': 'bold'})
])

# --------------------
# THE EFFICIENT CALLBACK
# --------------------
@app.callback(
    Output('output-map', 'figure'),
    Output('summary-stats', 'children'),
    Input('run-button', 'n_clicks'),
    State('num-vehicles-input', 'value'),
    State('vehicle-capacity-input', 'value'),
    State('solver-strategy-dropdown', 'value')
)
def update_map(n_clicks, num_vehicles, vehicle_capacity, strategy):
    if n_clicks == 0:
        return go.Figure(layout={'title': 'Set parameters and click "Run Solver"'}), ""
    
    orders_df, warehouses_df, all_locations_df, error = prepare_data()
    if error: return go.Figure(layout={'title': 'Error'}), error

    total_demand = orders_df['Total_Demand'].sum()
    total_fleet_capacity = num_vehicles * vehicle_capacity
    if total_demand > total_fleet_capacity:
        return go.Figure(layout={'title': 'Error'}), f"Error: Insufficient capacity. Demand: {total_demand}, Capacity: {total_fleet_capacity}"

    distance_matrix = np.array([[calculate_distance_in_miles(r1.lat, r1.lon, r2.lat, r2.lon) for _, r2 in all_locations_df.iterrows()] for _, r1 in all_locations_df.iterrows()])
    data = {'distance_matrix': distance_matrix, 'demands': [0]*len(warehouses_df) + list(orders_df['Total_Demand']),
            'vehicle_capacities': [vehicle_capacity]*num_vehicles, 'num_vehicles': num_vehicles, 'depot': 0}
    
    if strategy == 'optimized':
        fig, dist = solve_with_or_tools(data, all_locations_df, orders_df, warehouses_df)
        if fig is None: return go.Figure(layout={'title': 'Error'}), dist
        summary = f"Optimized Total Distance: {dist:.2f} miles"
        return fig, summary
    
    elif strategy == 'greedy':
        fig_greedy, dist_greedy = solve_with_greedy(data, all_locations_df, orders_df, warehouses_df)
        fig_opt, dist_optimized = solve_with_or_tools(data, all_locations_df, orders_df, warehouses_df)
        summary = f"Optimized: {dist_optimized:.2f} miles vs. Non-Optimized: {dist_greedy:.2f} miles"
        return fig_greedy, summary

# --------------------
# START THE SERVER
# --------------------
if __name__ == '__main__':
    app.run(debug=True)