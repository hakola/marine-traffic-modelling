import numpy as np
import pandas as pd
import shapely.geometry
import pyproj
from shapely.geometry import Point
import datetime


class Grid:

    def __init__(self, cols, rows, side_length, p_from, p_to):
        self.cols = cols
        self.rows = rows
        self.side_length = side_length
        self.p_from = p_from
        self.p_to = p_to
        self.model_1km = [
            [-1, -0],
            [-10, -1],
            [-10, -9],
            [-5, -1],
            [-3, -1],
            [-2, -1],
            [-1, -1],
            [0, -1],
            [-3, -2],
            [-1, -2],
            [-2, -3],
            [-1, -3],
            [-5, -4],
            [-4, -5],
            [-1, -5],
            [-9, -10],
            [-1, -10],
        ]
        self.model_2point5km = [
            [-1, -0],
            [-5, -1],
            [-3, -1],
            [-2, -1],
            [-1, -1],
            [0, -1],
            [-3, -2],
            [-1, -2],
            [-2, -3],
            [-1, -3],
            [-5, -4],
            [-4, -5],
            [-1, -5],
        ]
        self.model_5km = [
            [-1, -0],
            [-3, -1],
            [-2, -1],
            [-1, -1],
            [0, -1],
            [-3, -2],
            [-1, -2],
            [-2, -3],
            [-1, -3],
        ]

        if side_length < 2500:
            self.neighbor_model = self.model_1km
        elif 2500 <= side_length < 5000:
            self.neighbor_model = self.model_2point5km
        else:
            self.neighbor_model = self.model_5km

    def print_parameters(self):
        print('side_length:', self.side_length)


# Code and decode (row, col) pairs into integers, to represent the index of the node in graph
    def get_node_index(self, row, col):
        return (row * len(self.rows)) + col

    # Decode node index back to (row, col) pair
    def extract_coords(self, node):
        row = int((node / len(self.rows)))
        col = int(node - (row * len(self.rows)))

        return [row, col]

    # Decode node index back to (lat, lon) pair
    def extract_coords_lat_lon(self, node):
        row = int((node / len(self.rows)))
        col = int(node - (row * len(self.rows)))

        p = pyproj.transform(self.p_to, self.p_from, self.cols[col] + self.side_length/2, self.rows[row] + self.side_length/2)
        # p = pyproj.transform(self.p_to, self.p_from, self.cols[col-1] + self.side_length/2, self.rows[row-1] + self.side_length/2)

        return [p[1], p[0]]

    def get_neighbours_adjacent(self, row, col):
        neighbours = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i == 0) & (j == 0):
                    continue
                if (col+j >= 0) & (row+i >= 0) & (col+j < len(self.cols) - 1) & (row+i < len(self.rows) - 1):
                    neighbours.append([row+i , col+j])
        return neighbours


    def get_neighbours_with_maneuvaribility(self, row, col):
        neighbours = []
        # https://ieeexplore-ieee-org.libproxy.tuni.fi/mediastore_new/IEEE/content/media/6845395/6851348/6851512/6851512-fig-2-source-large.gif
        for i in [1, -1]:
            for j in [1, -1]:
                for index in self.neighbor_model:
                    x = index[0] * i
                    y = index[1] * j

                    if (col+y >= 0) & (row+x >= 0) & (col+y < len(self.cols) - 1) & (row+x < len(self.rows) - 1):
                        neighbours.append([row+x , col+y])
        return neighbours

    def get_neighbours_nodes_with_maneuvaribility(self, row, col):
        neighbours = []
        # https://ieeexplore-ieee-org.libproxy.tuni.fi/mediastore_new/IEEE/content/media/6845395/6851348/6851512/6851512-fig-2-source-large.gif
        for i in [1, -1]:
            for j in [1, -1]:
                for index in self.neighbor_model:
                    x = index[0] * i
                    y = index[1] * j

                    if (col+y >= 0) & (row+x >= 0) & (col+y < len(self.cols) - 1) & (row+x < len(self.rows) - 1):
                        neighbours.append(self.get_node_index(row+x, col+y))
        return neighbours

    def get_grid_position(self, row):
        return self.get_node_index(np.searchsorted(self.rows, row['grid_point'][1]) - 1, np.searchsorted(self.cols, row['grid_point'][0]) - 1)
        # return {
        #     'col': np.searchsorted(self.cols, row['grid_point'][0]),
        #     'row': np.searchsorted(self.rows, row['grid_point'][1])
        # }

    def get_grid_point(self, lat, lon):
        return pyproj.transform(self.p_from, self.p_to, lon, lat)


class Graph:

    def __init__(self):
        self.nodes = []


def area_to_grid(north_east=shapely.geometry.Point(31.5, 66.8), south_west=shapely.geometry.Point(8.5, 53), side_length=5000, proj_from='epsg:4326', proj_to='EPSG:3067'):
    # Set up projections
    p_from = pyproj.Proj(init=proj_from)
    p_to = pyproj.Proj(init=proj_to)# http://epsg.io/3067 #epsg:3857 metric; same as EPSG:900913

    # Create corners of rectangle to be transformed to a grid

    print('Original NE:', north_east)
    print('Original SW:', south_west)

    # Project corners to projection that is metric
    ne = pyproj.transform(p_from, p_to, north_east.x, north_east.y) # Transform NW point to 3067
    sw = pyproj.transform(p_from, p_to, south_west.x, south_west.y) # .. same for SE

    print('Transformed NE:', ne)
    print('Transformed SW:', sw)

    # Create grid cols and rows
    col_count = int((ne[0] - sw[0])/side_length)
    row_count = int((ne[1] - sw[1])/side_length)

    print('col_count', col_count)
    print('row_count', row_count)

    cols = np.linspace(sw[0], ne[0], num=col_count)
    rows = np.linspace(sw[1], ne[1], num=row_count)

    print('grid_size', (col_count*row_count))

    return Grid(cols, rows, side_length, p_from, p_to)


def get_node_avg_speeds(node, types, ais, grid):
    pos = grid.extract_coords(node)
    node_avg_speeds = {}
    for type in types:
        grid_pos = {'col': pos[1], 'row': [0]}
        node_avg_speeds[type] = ais[(ais.grid_pos == grid_pos) & (ais.vessel_type == type)].speed.mean()

    return node_avg_speeds


def get_avg_speeds_wo_graph(ais):
    types = ais.vessel_type.drop_duplicates().reset_index(drop=True).values
    nodes = ais.original.drop_duplicates().to_frame().reset_index(drop=True)
    nodes['avg_speeds'] = nodes.apply(lambda row: get_node_avg_speeds(row, types, ais))
    return nodes


def get_avg_speeds(ais):
    ais = ais[['vessel_type', 'node', 'speed']]
    groups = ais.groupby('node')

    avg_speeds = {}
    avg_speeds_arr = []
    for node, group in groups:
        node_avg_speeds = {}
        types = group.groupby('vessel_type')

        for vessel_type, positions in types:
            node_avg_speeds[vessel_type] = positions.speed.mean()

        avg_speeds[node] = node_avg_speeds
        avg_speeds_arr.append([node, node_avg_speeds])
    return pd.DataFrame(avg_speeds)


# Connect all the neighbour nodes in grid
def create_area_graph(grid, adjacent=False):
    graph = []
    for row in range(len(grid.rows) - 1):
        for col in range(len(grid.cols) - 1):

            node = grid.get_node_index(row, col)
            if adjacent:
                neighbours = grid.get_neighbours_adjacent(row, col)
            else:
                neighbours = grid.get_neighbours_with_maneuvaribility(row, col)
            for n in neighbours:
                connected = grid.get_node_index(n[0], n[1])
                graph.append([node, connected])

    return graph


# Connect all the neighbour nodes in grid
def create_sea_graph(grid, adjacent=False, exclude_geom=None):
    graph = []
    for row in range(len(grid.rows) - 1):
        for col in range(len(grid.cols) - 1):
            # Create point from col and row

            sw = shapely.geometry.Point(pyproj.transform(grid.p_to, grid.p_from, grid.cols[col], grid.rows[row]))
            se = shapely.geometry.Point(pyproj.transform(grid.p_to, grid.p_from, grid.cols[col] + grid.side_length, grid.rows[row]))
            nw = shapely.geometry.Point(pyproj.transform(grid.p_to, grid.p_from, grid.cols[col], grid.rows[row] + grid.side_length))
            ne = shapely.geometry.Point(pyproj.transform(grid.p_to, grid.p_from, grid.cols[col] + grid.side_length, grid.rows[row] + grid.side_length))
            pointlist = list([sw, se, ne, nw])

            cell = shapely.geometry.Polygon([[p.x, p.y] for p in pointlist])

            if exclude_geom is not None and exclude_geom.contains(cell):
                continue
            node = grid.get_node_index(row, col)
            if adjacent:
                neighbours = grid.get_neighbours_adjacent(row, col)
            else:
                neighbours = grid.get_neighbours_with_maneuvaribility(row, col)
            for n in neighbours:
                connected = grid.get_node_index(n[0], n[1])
                graph.append([node, connected])

    return graph


def get_port_lat(ports, row):
    return ports.loc[row.port_id].lat


def get_port_lon(ports, row):
    return ports.loc[row.port_id].lon


def get_vessel_type(vessels, row):
    return vessels.loc[row.mmsi].vessel_type


def get_port_area(port_areas, row):
    i = 0
    for area in port_areas:
        p = Point(row.port_lon, row.port_lat)
        if area.contains(p):
            return i
        i += 1

    return None


def create_graph_from_ais_adjacent(ais):
    ais = ais.reset_index(drop=True)
    ais = ais.sort_values(by=['mmsi', 'timestamp'])

    observations_by_vessels = ais.groupby('mmsi')
    graph = []

    for mmsi, observations in observations_by_vessels:
        prev_node = observations.head(1).iloc[0].node
        for i, obs in observations.iterrows():
            if prev_node == obs.node:
                continue

            node_row = [prev_node, obs.node]
            graph.append(node_row)

            prev_node = obs.node

    return graph


def create_graph_from_ais_maneuvaribility(grid, ais, minutes_forward=45):
    ais = ais.reset_index(drop=True)
    ais = ais.sort_values(by=['mmsi', 'timestamp'])

    observations_by_vessels = ais.groupby('mmsi')

    graph = []

    for mmsi, observations in observations_by_vessels:
        observations = observations.set_index('timestamp')
        node = None
        for i, obs in observations.iterrows():
            if node == obs.node:
                continue

            node = obs.node
            coords = grid.extract_coords(node)

            neighbors = grid.get_neighbours_nodes_with_maneuvaribility(coords[0], coords[1])
            obs_time = pd.to_datetime(obs.name)
            time_window = observations.loc[obs.name : (pd.Timestamp(obs_time + datetime.timedelta(minutes=minutes_forward)))]
            future_nodes = time_window.node.unique()
            future_nodes_set = set(future_nodes)

            for neighbor in neighbors:
                if neighbor in future_nodes_set:
                    graph.append([node, neighbor])

    return graph


# Function for calculating weights for the graph. Could be improved a lot.
def calculate_transition_cost(sea_graph, traffic_graph):
    weight_graph = traffic_graph.groupby(['original', 'connected']).size()
    weight_graph = weight_graph.to_frame().reset_index()
    weight_graph.columns = ['original', 'connected', 'occurences']
    weight_graph = weight_graph.set_index(['original', 'connected'])

    max_occ = weight_graph.occurences.max()

    sea_graph['weight'] = sea_graph.apply(get_normal_weights, args=(weight_graph, max_occ), axis=1)
    print('Weights calculated')

    max_weight = sea_graph.weight.max()
    sea_graph['cost'] = sea_graph.apply(lambda row: row.weight/max_weight, axis=1)
    print('Costs calculated')
    return sea_graph


def get_normal_weights(row, weight_graph, max_occ):
    max_weight = max_occ * 5
    try:
        weight = max_occ - weight_graph.loc[(row.original, row.connected), 'occurences']
        return weight
    except:
        return max_weight
