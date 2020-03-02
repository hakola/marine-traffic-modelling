from collections import defaultdict
from sklearn import preprocessing
import pandas as pd
from math import *
import datetime
#from .pygradu import portcalls
from .gridify import *
import math

EARTH_RADIUS_KM = 6371.0


class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.costs = {}
        self.type_weights = {}
        self.positions = {}
        self.use_dirways = True
        self.use_turn_penalty = False
        self.use_shallow_penalty = False

    def add_edge(self, from_node, to_node, cost):
        # Note: assumes edges are bi-directional
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)

        self.costs[(from_node, to_node)] = cost

    def cost(self, current_node, next_node, current_course, next_course, dirways_graph, shallow_graph):
        cost = self.costs[(current_node, next_node)]

        if self.use_dirways and next_node in dirways_graph:
            return 0.05

        shallow_penalty = 0
        if self.use_shallow_penalty and next_node in shallow_graph:
            shallow_penalty = 0.2

        turn_penalty = 0
        if self.use_turn_penalty:
            phi = abs(current_course - next_course) % 360
            if phi > 180:
                change = 360 - phi
            else:
                change = phi
            turn_penalty = change/180 * 0.05

        return cost + turn_penalty + shallow_penalty

    def print_parameters(self):
        print('use_dirways=', self.use_dirways)
        print('use_turn_penalty=', self.use_turn_penalty)
        print('use_shallow_penalty=', self.use_shallow_penalty)



def df_to_graph(complete_graph):
    edges = complete_graph.values

    graph = Graph()
    # Convert df to format that A* understands
    for e in edges:
        edge = (e[0], e[1], e[2])
        graph.add_edge(*edge)

    return graph

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None, coords=None, speed=None, course=None, transitions=None):
        self.parent = parent
        self.position = position
        self.coords = coords

        self.g = 0
        self.h = 0
        self.f = 0

        self.speed = speed
        self.course = course
        self.transitions = transitions

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)


def deg2rad(deg):
    return deg * (math.pi / 180)


def rad2deg(rad):
    return rad * (180 / math.pi)


def normalize(value, min, max):
    if value < min:
        return value + (max - min)
    if value > max:
        return value - (max - min)
    return value


def angleFromCoordinatesInDeg(coordinate1, coordinate2):
    lat1 = deg2rad(coordinate1[0])
    lat2 = deg2rad(coordinate2[0])
    long1 = deg2rad(coordinate1[1])
    long2 = deg2rad(coordinate2[1])
    dLon = (long2 - long1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    bearing = math.atan2(y, x)
    return normalize(rad2deg(bearing), 0.0, 360.0)


def distance_from_coords_in_km(coordinate1, coordinate2):
    lat1 = deg2rad(coordinate1[0])
    lat2 = deg2rad(coordinate2[0])
    long1 = deg2rad(coordinate1[1])
    long2 = deg2rad(coordinate2[1])

    dLat = (lat2 - lat1)
    dLon = (long2 - long1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos((lat1)) * math.cos((lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def calculate_time(coords1, coords2, speed, start_time):
    distance = distance_from_coords_in_km(coords1, coords2) * 1000
    return start_time + datetime.timedelta(seconds=distance / speed)


def interpolate_to_time(latlon1, latlon2, speed, start_time, end_time):
    bearing = deg2rad(angleFromCoordinatesInDeg(latlon1, latlon2))

    lat1 = deg2rad(latlon1[0])
    long1 = deg2rad(latlon1[1])

    distance = ((end_time-start_time).total_seconds() * speed) / 1000

    lat2 = math.asin(math.sin(lat1) * math.cos(distance / EARTH_RADIUS_KM) + math.cos(lat1) * math.sin(distance / EARTH_RADIUS_KM) * math.cos(bearing))

    long2 = long1 + math.atan2(math.sin(bearing) * math.sin(distance / EARTH_RADIUS_KM) * math.cos(lat1), math.cos(distance / EARTH_RADIUS_KM) - math.sin(lat1) * math.sin(lat2))

    return [rad2deg(lat2), rad2deg(long2)]


def get_speed(avg_speeds, vessel_type, prev_speed, node_pos, transitions):
    max_transitions = 3
    if transitions > max_transitions:
        transitions = max_transitions
    multiplier = max_transitions - transitions
    # Speed as moving average
    try:
        speeds_by_type = avg_speeds[node_pos]
        pred_speed = speeds_by_type[vessel_type]
        if isnan(speeds_by_type[vessel_type]):
            pred_speed = 7
    except KeyError:
        pred_speed = 7

    speed = ((prev_speed * multiplier) + (pred_speed * (max_transitions - multiplier))) / max_transitions

    if speed > 5:
        speed += 0.28

    return speed


def pythagoras(x,y):
    return math.sqrt(((x.coords[0] - y.coords[0]) ** 2) + ((x.coords[1] - y.coords[1]) ** 2))


# TODO: Calculate timestamps for each node instead of adding the start time into them
def retrace_route(grid, current_node, start_latlon, end_latlon, mmsi, voyage, start_time):
    path = []
    current = current_node

    row = []
    while current is not None:
        if len(path) is 0:
            row = end_latlon
        elif current.parent is None:
            row = start_latlon
        else:
            row = grid.extract_coords_lat_lon(current.position)

        row.extend([current.position, current.speed, mmsi, voyage, start_time, current.transitions])
        path.append(row)
        current = current.parent
    return path[::-1]  # Return reversed pat


def retrace_search_area(grid, closed_list, voyage):
    area = []
    for node in closed_list:
        row = grid.extract_coords_lat_lon(node.position)
        row.extend([voyage, node.g, node.h, node.f])
        area.append(row)

    return area


def distance_to_dest(next_node, end_node, speed):
    return pythagoras(next_node, end_node)


def manhattan_distance(x,y, speed):
    return sum(abs(a-b) for a, b in zip(x.coords, y.coords))


def diagonal_distance(next_node, end_node, speed):
    d = 1
    d2 = sqrt(2)
    dx = abs(next_node.coords[0] - end_node.coords[0])
    dy = abs(next_node.coords[1] - end_node.coords[1])
    return (d * (dx + dy) + (d2 - 2 * d) * min(dx, dy))


def a_star(graph, start_latlon, end_latlon, avg_speeds, speed, course, vessel_type, grid, dirways_graph, shallow_graph, mmsi, voyage, start_time):
    # open_list is a list of nodes which have been visited, but who's neighbors
    # haven't all been inspected, starts off with the start node
    # closed_list is a list of nodes which have been visited
    # and who's neighbors have been inspected

    open_list = set()
    closed_list = set()

    start_pos = get_node(grid, start_latlon[0], start_latlon[1])
    end_pos = get_node(grid, end_latlon[0], end_latlon[1])

    start_node = Node(None, start_pos, grid.extract_coords(start_pos), speed, course, 0)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end_pos, grid.extract_coords(end_pos))
    end_node.g = end_node.h = end_node.f = 0

    # parents contains an adjacency map of all nodes
    parents = dict()
    parents[start_node.position] = start_node

    open_list.add(start_node)

    while open_list:
        current_node = None

        # find a node with the lowest value of f() - evaluation function
        for tmp_node in open_list:
            if current_node is None or tmp_node.f < current_node.f:
                current_node = tmp_node

        if current_node is None:

            print('Failed to predict route')
            return None

        # Found the goal
        if current_node == end_node:
            route = retrace_route(grid, current_node, start_latlon, end_latlon, mmsi, voyage, start_time)
            search_area = retrace_search_area(grid, closed_list, voyage)
            return [route, search_area]

        neighbours = graph.edges[current_node.position]

        # for all neighbors of the current node do
        for next_node in neighbours:
            speed = get_speed(avg_speeds, vessel_type, current_node.speed, current_node.position, current_node.transitions + 1)
            course = None

            if graph.use_turn_penalty:
                current_latlon = grid.extract_coords_lat_lon(current_node.position)
                next_latlon = grid.extract_coords_lat_lon(next_node)
                course = angleFromCoordinatesInDeg(current_latlon, next_latlon)

            next_node = Node(current_node, next_node, grid.extract_coords(next_node), speed, course, current_node.transitions + 1)

            if next_node in closed_list:
                continue

            # G is the sum of all costs from the beginning
            next_node.g = current_node.g + graph.cost(current_node.position, next_node.position, current_node.course, next_node.course, dirways_graph, shallow_graph)
            next_node.h = diagonal_distance(next_node, end_node, speed)
            next_node.f = next_node.g + next_node.f

            # if the current node isn't in both open_list and closed_list
            # add it to open_list and note n as it's parent
            if next_node not in open_list:
                open_list.add(next_node)
            else:
                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                for open_neighbor in open_list:
                    if open_neighbor == next_node and next_node.g < open_neighbor.g:
                        open_neighbor.g = next_node.g
                        open_neighbor.h = next_node.h
                        open_neighbor.f = next_node.f
                        open_neighbor.parent = next_node.parent
                        open_neighbor.transitions = next_node.transitions
                        open_neighbor.speed = next_node.speed
                        open_neighbor.course = next_node.course

                        if next_node in closed_list:
                            closed_list.remove(next_node)
                            open_list.add(next_node)

        # remove n from the open_list, and add it to closed_list
        # because all of his neighbors were inspected
        open_list.remove(current_node)
        closed_list.add(current_node)

    print('Path does not exist!')
    print('Voyage=', str(voyage))
    return None


def measure_accuracy(grid, real_pos, pred_pos, pred_speed, actual_speed):
    nm_multiplier = 0.539956803
    distance_nm = distance_from_coords_in_km(real_pos, pred_pos) * nm_multiplier

    error_rate_lat = abs(real_pos[0] - pred_pos[0]) / real_pos[0] * 100
    error_rate_lon = abs(real_pos[1] - pred_pos[1]) / real_pos[1] * 100

    error_rate_speed = None
    if actual_speed > 0:
        error_rate_speed = abs((actual_speed - pred_speed)) / actual_speed * 100

    real_node = get_node(grid, real_pos[0], real_pos[1])
    pred_node = get_node(grid, pred_pos[0], pred_pos[1])

    return [distance_nm, error_rate_lat, error_rate_lon, error_rate_speed, int(real_node == pred_node)]


def extract_test_voyage_ids(voyages, port_id, n):
    voyage_sizes = voyages.loc[voyages['port_id'] == port_id].loc[voyages['speed'] > 2].groupby([ 'voyage']).size().sort_values(ascending=False)
    return voyage_sizes.head(n).index.values.tolist()


def get_test_voyage(voyages, voyage_id, minutes_forward):
    test_voyage = voyages[voyages.voyage == voyage_id]
    ata = test_voyage.head(1).iloc[0].ata
    start_time = pd.to_datetime(test_voyage.head(1).iloc[0].timestamp)

    if test_voyage.head(1).iloc[0].speed < 1:
        return None

    test_voyage = test_voyage.loc[test_voyage['timestamp'] <= ata]
    test_voyage = test_voyage.loc[test_voyage['timestamp'] <= pd.Timestamp(start_time + datetime.timedelta(minutes=minutes_forward))]
    return test_voyage


def get_node(grid, lat, lon):
    start = dict()
    start['grid_point'] = grid.get_grid_point(lat, lon)
    return grid.get_grid_position(start)


def interpolate_by_distance(row, next_row, distanceKm):
    brng = deg2rad(angleFromCoordinatesInDeg([row.lat, row.lon], [next_row.lat, next_row.lon]))
    lat1 = deg2rad(row.lat)
    long1 = deg2rad(row.lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(distanceKm / EARTH_RADIUS_KM) + math.cos(lat1) * math.sin(distanceKm / EARTH_RADIUS_KM) * math.cos(brng))
    long2 = long1 + math.atan2(math.sin(brng) * math.sin(distanceKm / EARTH_RADIUS_KM) * math.cos(lat1), math.cos(distanceKm / EARTH_RADIUS_KM) - math.sin(lat1) * math.sin(lat2))

    row.lat = rad2deg(lat2)
    row.lon = rad2deg(long2)

    return row


def create_dirways_graph(dirways, grid):
    dirways.sort_values(by=['id', 'number'], inplace=True)
    dirways.reset_index(inplace=True)

    dirways = dirways.groupby('id')

    interpolated_dirways = []
    distance_km = 1

    dirway_nodes = set()

    for id, dw_points in dirways:
        order_number = 0
        interpolated = []
        dw_points.reset_index(inplace=True)
        for i, current in dw_points.iterrows():
            interpolated.append(current)

            if i+1 == len(dw_points.lat):
                break

            next = dw_points.loc[(i+1)]
            current.number = order_number

            coords = dict()
            coords['grid_point'] = grid.get_grid_point(current.lat, current.lon)

            node = grid.get_grid_position(coords)

            if node not in dirway_nodes:
                dirway_nodes.add(node)

            while distance_km < distance_from_coords_in_km([current.lat, current.lon], [next.lat, next.lon]):
                current = interpolate_by_distance(current, next, distance_km)
                order_number += 1
                current.number = order_number
                interpolated.append(current)

                coords = dict()
                coords['grid_point'] = grid.get_grid_point(current.lat, current.lon)

                node = grid.get_grid_position(coords)

                if node not in dirway_nodes:
                    dirway_nodes.add(node)

        interpolated_dirways.append(interpolated)

    return dirway_nodes


def get_observations_at_time(voyages, timestamp):
    start_time = pd.to_datetime(timestamp)
    voyages['course'] = -1
    columns = voyages.columns
    voyages = voyages.loc[(voyages['timestamp'] >= timestamp) & (voyages['timestamp'] < pd.Timestamp(start_time + datetime.timedelta(minutes=60))) & (voyages['ata'] > timestamp) & (voyages['speed'] > 1)]

    voyages = voyages.groupby('voyage')
    test_voyages = []
    for voyage, observations in voyages:
        if len(observations) is 1:
            continue
        course = angleFromCoordinatesInDeg([observations.iloc[0].lat, observations.iloc[0].lon], [observations.iloc[1].lat, observations.iloc[1].lon])
        row = observations.iloc[1]
        row.course = course
        test_voyages.append(row)

    return pd.DataFrame(data=test_voyages, columns=columns)


def predict_routes(observations, grid, graph, avg_speeds, dirways, shallow_graph, print_params=True):
    if print_params:
        graph.print_parameters()
        grid.print_parameters()

    routes = []
    errors = []
    search_areas = []
    for i, observation in observations.iterrows():
        dirway_graph = None
        if graph.use_dirways:
            active_dirways = dirways.loc[(dirways.publishtime < observation.ata) & (observation.ata <= dirways.deletetime)]
            dirway_graph = create_dirways_graph(active_dirways, grid)

        start_coords = [observation.lat, observation.lon]
        start_time = pd.to_datetime(observation.timestamp)

        end_coords = [observation.end_lat, observation.end_lon]

        route = a_star(graph, start_coords, end_coords, avg_speeds, observation.speed, observation.course, observation.vessel_type, grid
                       , dirway_graph, shallow_graph, observation.mmsi, observation.voyage, start_time)
        if route is None:
            errors.append([start_coords,  end_coords])
        else:
            routes.extend(route[0])
            search_areas.extend(route[1])
    # return routes
    if print_params:
        print('Error count=',len(errors))
        print(errors)
    return [routes, search_areas]


# def calculate_time(coords1, coords2, speed, start_time):

def calculate_timestamps(routes):
    routes.sort_values(by=['voyage', 'number'], inplace=True)
    routes['timestamp'] = -1
    routes = routes.groupby('voyage')

    test = []
    for voyage, route in routes:
        route = route.reset_index(drop=True)
        for i, current in route.iterrows():
            timestamp = None

            if i == 0:
                timestamp = current.start_time
            else:
                current.speed = (prev.speed+current.speed) / 2
                timestamp = calculate_time([prev.lat, prev.lon], [current.lat, current.lon], current.speed, prev.timestamp)

            current.timestamp = timestamp
            prev = current
            test.append(current.values)

    return pd.DataFrame(data=test, columns=['lat', 'lon', 'node', 'speed', 'mmsi', 'voyage', 'start_time', 'number', 'timestamp'])


def test_accuracy(grid, predicted, voyages, minutes_forward=None):
    predicted.sort_values(by=['voyage', 'number'], inplace=True)
    predicted = predicted.groupby('voyage')
    results = []
    errors = []
    for voyage, pred_route in predicted:
        start_row = pred_route.head(1).iloc[0]
        end_row = pred_route.tail(1).iloc[0]
        actual_route = voyages.loc[(voyages['voyage'] == voyage) & (voyages['timestamp'] >= start_row.start_time) & (voyages['timestamp'] <= end_row.timestamp)]
        pred_route.sort_values(by=['timestamp'], inplace=True)
        pred_route = pred_route.reset_index(drop=True)

        for i, obs in actual_route.iterrows():
            try:
                if minutes_forward is not None and pd.to_datetime(obs.timestamp) > (pd.to_datetime(start_row.start_time) + datetime.timedelta(minutes=minutes_forward)):
                    break
                next = pred_route.loc[pred_route['timestamp'] > obs.timestamp].head(1).iloc[0]
                next_index = next.name
                pred_pos = None
                pred_speed = None
                if int(next_index) is 0:
                    pred_pos = [pred_route.head(1).iloc[0].lat, pred_route.head(1).iloc[0].lon]
                    pred_speed = pred_route.head(1).iloc[0].speed
                else:
                    prev = pred_route.loc[next_index-1]
                    pred_speed = (prev.speed + next.speed) / 2
                    pred_pos = interpolate_to_time([prev.lat, prev.lon], [next.lat, next.lon], pred_speed, prev.timestamp, obs.timestamp)

                mins_to_future = (pd.to_datetime(obs.timestamp) - pd.to_datetime(start_row.start_time)).total_seconds() / 60.0

                acc_measures = measure_accuracy(grid, [obs.lat, obs.lon], pred_pos, pred_speed, obs.speed)
                result = [voyage, obs.vessel_type, obs.end_port, obs.end_port_sea_area, start_row.start_time, obs.timestamp, mins_to_future, obs.lat, obs.lon, pred_pos[0], pred_pos[1], obs.speed, pred_speed]
                result.extend(acc_measures)
                results.append(result)
            except IndexError:
                errors.append(obs)
                continue

    print('error count=', len(errors))
    columns = ['voyage', 'vessel_type', 'end_port', 'end_port_sea_area', 'start_time', 'pred_time', 'mins_to_future', 'actual_lat', 'actual_lon', 'pred_lat',
               'pred_lon', 'actual_speed', 'pred_speed', 'acc_distance_nm', 'error_rate_lat', 'error_rate_lon', 'error_rate_speed', 'correct_node']
    return pd.DataFrame(data=results, columns=columns)


def test_accuracy_to_end(grid, predicted, voyages, minutes_forward=None):
    predicted.sort_values(by=['voyage', 'number'], inplace=True)
    predicted = predicted.groupby('voyage')
    results = []
    errors = []
    for voyage, pred_route in predicted:
        start_row = pred_route.head(1).iloc[0]
        end_row = pred_route.tail(1).iloc[0]
        actual_route = voyages.loc[(voyages['voyage'] == voyage) & (voyages['timestamp'] >= start_row.start_time) & (voyages['timestamp'] <= end_row.timestamp)]
        actual_end = actual_route.tail(1).iloc[0]
        pred_route.sort_values(by=['timestamp'], inplace=True)
        pred_route = pred_route.reset_index(drop=True)
        prev = None
        last = None
        for i, next in pred_route.iterrows():
            last = next
            if prev is None:
                prev = next
                continue

            observations_between_preds = actual_route.loc[(actual_route['timestamp'] >= prev.timestamp) & (actual_route['timestamp'] < next.timestamp)]
            pred_speed = (prev.speed + next.speed) / 2

            if actual_end.ata < next.timestamp:
                mins_to_future = (pd.to_datetime(next.timestamp) - pd.to_datetime(start_row.start_time)).total_seconds() / 60.0
                acc_measures = measure_accuracy(grid, [actual_end.lat, actual_end.lon], [next.lat, next.lon], pred_speed, actual_end.speed)
                result = [voyage, actual_end.vessel_type, actual_end.end_port, actual_end.end_port_sea_area, start_row.start_time, next.timestamp, mins_to_future,
                          actual_end.lat, actual_end.lon, next.lat, next.lon, actual_end.speed, pred_speed]
                result.extend(acc_measures)
                results.append(result)
                prev = next
                continue

            for j, obs in observations_between_preds.iterrows():
                if minutes_forward is not None and pd.to_datetime(obs.timestamp) > (pd.to_datetime(start_row.start_time) + datetime.timedelta(minutes=minutes_forward)):
                    break
                pred_pos = interpolate_to_time([prev.lat, prev.lon], [next.lat, next.lon], pred_speed, prev.timestamp, obs.timestamp)
                mins_to_future = (pd.to_datetime(obs.timestamp) - pd.to_datetime(start_row.start_time)).total_seconds() / 60.0
                acc_measures = measure_accuracy(grid, [obs.lat, obs.lon], pred_pos, pred_speed, obs.speed)
                result = [voyage, obs.vessel_type, obs.end_port, obs.end_port_sea_area, start_row.start_time, obs.timestamp, mins_to_future, obs.lat, obs.lon, pred_pos[0], pred_pos[1], obs.speed, pred_speed]
                result.extend(acc_measures)
                results.append(result)

            prev = next

    print('error count=', len(errors))
    columns = ['voyage', 'vessel_type', 'end_port', 'end_port_sea_area', 'start_time', 'pred_time', 'mins_to_future', 'actual_lat', 'actual_lon', 'pred_lat',
               'pred_lon', 'actual_speed', 'pred_speed', 'acc_distance_nm', 'error_rate_lat', 'error_rate_lon', 'error_rate_speed', 'correct_node']
    return pd.DataFrame(data=results, columns=columns)


def plot_intervals(results, col_x, col_y, interval):
    bins = []
    labels = []

    i = 0
    while i < results[col_x].max():
        bins.append(i)
        labels.append(int(i))
        i += interval
    del labels[0]

    results['interval'] = pd.cut(results[col_x], bins=bins, labels=labels)

    x = []
    y = []

    for group, voyage_results in results.groupby('interval'):
        if(len(voyage_results)) < 1 | int(group/60) > 50:
            continue
        y.append(voyage_results[col_y].mean())
        if np.isnan(group):
            group = 0
        x.append(group/60)

    return x, y
