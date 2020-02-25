import math
import datetime
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
KNOTS_PER_MS = 1.943844
KM_PER_NM = 1.852
time_limit_hours = 2
min_distance_to_port = 10
not_available = -1


def deg2rad(deg):
    return deg * (math.pi / 180)


def rad2deg(rad):
    return rad * (180 / math.pi)


def distance_from_coords_in_km(coordinate1, coordinate2):
    lat1 = deg2rad(coordinate1['lat'])
    lat2 = deg2rad(coordinate2['lat'])
    long1 = deg2rad(coordinate1['lon'])
    long2 = deg2rad(coordinate2['lon'])

    dLat = (lat2 - lat1)
    dLon = (long2 - long1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos((lat1)) * math.cos((lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def get_port_id(row, ports):
    row['lat'] = row['start_lat']
    row['lon'] = row['start_lon']
    for i, port in ports.iterrows():
        if distance_from_coords_in_km(row, port) < min_distance_to_port:
            return int(port['port_id'])

    return None


def get_port_id_new(row, ports, columns):
    row_coord = dict()
    row_coord['lat'] = row[columns.index('lat')]
    row_coord['lon'] = row[columns.index('lon')]
    for i, port in ports.iterrows():
        if distance_from_coords_in_km(row_coord, port) < min_distance_to_port:
            return port['port_id']

    return None


def is_moving(row):
    return row['speed'] > 0


def is_stopped(row):
    return row['speed'] < 0.2


def reset_voyage(voyage_nro):
    return {
        'end_port': not_available,
        'start_port': not_available,
        'end_lat': not_available,
        'end_lon': not_available,
        'ata': not_available,
        'atd': not_available,
        'voyage_nro': voyage_nro,
        'rows': []
    }


def voyage_to_array(voyage, columns):
    np_voyage = voyage['rows']
    np_voyage[:, columns.index('end_port')] = voyage['end_port']
    np_voyage[:, columns.index('ata')] = voyage['ata']
    np_voyage[:, columns.index('atd')] = voyage['atd']
    np_voyage[:, columns.index('start_port')] = voyage['start_port']
    np_voyage[:, columns.index('voyage')] = voyage['voyage_nro']
    return np_voyage


def voyage_to_df(voyage, columns):
    df_voyage = pd.DataFrame(data=voyage['rows'], columns=columns)
    df_voyage['end_port'] = voyage['end_port']
    df_voyage['ata'] = voyage['ata']
    df_voyage['atd'] = voyage['atd']
    df_voyage['end_lat'] = voyage['end_lat']
    df_voyage['end_lon'] = voyage['end_lon']
    df_voyage['start_port'] = voyage['start_port']
    df_voyage['voyage'] = voyage['voyage_nro']
    return df_voyage


def calculate_voyage_distance(voyage):
    voyage = voyage.sort_values(by=['timestamp'])
    voyage = voyage.reset_index(drop=True)

    distance = 0

    for i, row in voyage.iterrows():
        cursor = i+1
        if cursor == len(voyage.lat):
            continue
        next_row = voyage.loc[cursor]

        distance += distance_from_coords_in_km(row, next_row)

    return distance


def reset_ais_for_portcalls(ais):
    ais = ais[['timestamp', 'mmsi', 'lat', 'lon', 'speed', 'node', 'vessel_type', 'iceclass']]
    ais = ais.sort_values(by=['mmsi', 'timestamp'])
    ais = ais.reset_index(drop=True)
    ais['start_port'] = not_available
    ais['end_port'] = not_available
    ais['end_lat'] = not_available
    ais['end_lon'] = not_available
    ais['ata'] = not_available
    ais['atd'] = not_available
    ais['in_port'] = not_available
    ais['voyage'] = not_available
    return ais


def calculate_voyages(ais, ports):
    ais = reset_ais_for_portcalls(ais)
    vessels = ais.groupby('mmsi')

    ais_portcalls = pd.DataFrame(data=[], columns=ais.columns)

    voyage = reset_voyage(0)
    cursor = -1

    for mmsi, observations in vessels:
        voyage = reset_voyage(voyage['voyage_nro'])
        prev = observations.head(1).iloc[0]
        prev.in_port = False
        observations['long_speed'] = observations.speed.iloc[::-1].rolling(3).mean().iloc[::-1]
        observations['really_long_speed'] = observations.speed.iloc[::-1].rolling(5).mean().iloc[::-1]
        for i, obs in observations.iterrows():
            cursor += 1
            obs.in_port = obs.node in ports

            ais_fresh = pd.to_datetime(obs['timestamp']) <= (pd.to_datetime(prev['timestamp']) + datetime.timedelta(hours=time_limit_hours))

            if not ais_fresh:
                voyage = reset_voyage(voyage['voyage_nro'])
                prev = obs
                continue

            if voyage['atd'] == not_available and prev.in_port \
                    and not obs.in_port and obs.long_speed > 0:
                voyage['atd'] = obs.timestamp
                voyage['start_port'] = ports[prev.node]

            if voyage['atd'] != not_available:
                voyage['rows'].append(obs)

            if voyage['atd'] != not_available and obs.in_port \
                    and ports[obs.node] != voyage['start_port'] and obs.long_speed < 0.1:
                voyage['ata'] = obs.timestamp
                voyage['end_port'] = ports[obs.node]
                voyage['end_lat'] = obs.lat
                voyage['end_lon'] = obs.lon

                df_voyage = voyage_to_df(voyage, ais.columns)
                ais_portcalls = ais_portcalls.append(df_voyage, ignore_index = True)
                voyage = reset_voyage(voyage['voyage_nro'] + 1)
                prev = obs
                continue

            if voyage['atd'] != not_available and obs.in_port \
                    and ports[obs.node] == voyage['start_port']:
                voyage = reset_voyage(voyage['voyage_nro'])
                prev = obs
                continue

            if voyage['atd'] != not_available and obs.really_long_speed < 0.1:
                voyage = reset_voyage(voyage['voyage_nro'])
            prev = obs

            if cursor % 500000 == 0:
                print('i=' + str(i))
                ais_portcalls.to_csv('ais_portcalls' + str(i) + '.csv')

    return ais_portcalls


def get_voyage_lengths(ais):
    voyages = ais.groupby('voyage')
    rows = []
    remove_ids = []
    for i, voyage in voyages:
        voyage_dist = calculate_voyage_distance(voyage)
        start = voyage.head(1).iloc[0]
        end = voyage.tail(1).iloc[0]
        start_end_dist = distance_from_coords_in_km(start, end)

        travel_time = (pd.to_datetime(end.timestamp) - pd.to_datetime(start.timestamp)).total_seconds() / 60.0

        mean_speed = voyage.speed.mean()

        if start_end_dist == 0:
            remove_ids.append(i)
            continue
        difference = ((voyage_dist-start_end_dist)/start_end_dist) * 100

        rows.append([i, voyage_dist, start_end_dist, difference, travel_time, mean_speed])



    return pd.DataFrame(data=rows, columns=['voyage', 'voyage_length', 'start_end_length', 'difference', 'travel_time', 'avg_speed'])


def get_voyage_total_accuracy(pred_voyages, actual_voyages):
    pred_voyages = pred_voyages.groupby('voyage')
    rows = []
    for i, pred_voyage in pred_voyages:
        pred_voyage_dist = portcalls.calculate_voyage_distance(pred_voyage)

        pred_start = pred_voyage.head(1).iloc[0]
        pred_end = pred_voyage.tail(1).iloc[0]

        pred_travel_time = (pd.to_datetime(pred_end.timestamp) - pd.to_datetime(pred_start.timestamp)).total_seconds() / 60.0  / 60.0
        pred_mean_speed = pred_voyage.speed.mean()

        actual_voyage = actual_voyages[actual_voyages.voyage == i]
        actual_voyage_dist = portcalls.calculate_voyage_distance(actual_voyage)

        actual_start = actual_voyage.head(1).iloc[0]
        actual_end = actual_voyage.tail(1).iloc[0]

        actual_travel_time = (pd.to_datetime(actual_end.timestamp) - pd.to_datetime(actual_start.timestamp)).total_seconds() / 60.0 / 60.0
        actual_mean_speed = actual_voyage.speed.mean()

        dist_dif = abs(actual_voyage_dist-pred_voyage_dist)
        travel_time_dif = abs(actual_travel_time-pred_travel_time)
        mean_speed_dif = abs(actual_mean_speed-pred_mean_speed)

        actual_ata = actual_end.timestamp
        predicted_ata = pred_end.timestamp
        ata_diff_minutes = abs((pd.to_datetime(actual_ata) - pd.to_datetime(predicted_ata)).total_seconds()) / 60.0

        rows.append([i, actual_voyage_dist, pred_voyage_dist, dist_dif, actual_travel_time, pred_travel_time, travel_time_dif,
                     actual_mean_speed, pred_mean_speed, mean_speed_dif, actual_ata, predicted_ata, ata_diff_minutes])

    return pd.DataFrame(data=rows, columns=['voyage', 'actual_voyage_dist', 'pred_voyage_dist', 'dist_dif', 'actual_travel_time', 'pred_travel_time', 'travel_time_dif',
                                            'actual_mean_speed', 'pred_mean_speed', 'mean_speed_dif', 'actual_ata', 'pred_ata', 'ata_diff_minutes'])
