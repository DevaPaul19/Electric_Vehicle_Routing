import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import requests
from itertools import combinations
import numpy as np
import time
import sys
# Create an empty graph
graph = nx.Graph()
start_long=-123.3697010314036
start_lat=48.456943893386665
end_long=-123.37959459657861
end_lat=48.446957454018445
# Example start and end coordinates
start_coord = [start_long, start_lat]
end_coord = [end_long, end_lat]

# Add start and end coordinates as nodes
graph.add_node("Start", pos=start_coord)
graph.add_node("End", pos=end_coord)
max_lon,max_lat = -180, -90
min_lon, min_lat = 180, 90
charging_stations=[]

# Read charging station data from JSON file
df = pd.read_json('C:\\My folders\\Deva paul\\AI\\SEM 4\\IT - 257 DAA\\IT257_Project\\open\\charge_vic.json')

for i in range(df.shape[0]):
    coord = [df['AddressInfo'][i]['Longitude'], df['AddressInfo'][i]['Latitude']]
    charging_stations.append(coord)
    if coord[0] > max_lon:
        max_lon = coord[0]
    if coord[1] > max_lat:
        max_lat = coord[1]
    if coord[0] < min_lon:
        min_lon = coord[0]
    if coord[1] < min_lat:
        min_lat = coord[1]
print("max0 and max1",max_lon,max_lat)
print("min0 and min1",min_lon,min_lat)

# Define the bounding box
min_lat = min(start_coord[1], end_coord[1])
max_lat = max(start_coord[1], end_coord[1])
min_lon = min(start_coord[0], end_coord[0])
max_lon = max(start_coord[0], end_coord[0])

# Filter charging stations within the bounding box
filtered_charging_stations = []
for charging_station in charging_stations:
    lat = charging_station[1]
    lon = charging_station[0]
    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
        filtered_charging_stations.append(charging_station)


import folium
import polyline
for i in range(len(charging_stations)):
    charging_stations[i] = list(charging_stations[i])
map_coords =[]
for i in range(len(charging_stations)):
    a, b = charging_stations[i][0], charging_stations[i][1]
    temp=[]
    temp.append(b)
    temp.append(a)
    map_coords.append(temp)
# Extract the geometry string from the data
#geometry = json_data['routes'][0]['geometry']

# Decode the geometry string to get the latitude and longitude coordinates
#coordinates = map_coords#polyline.decode(geometry)
start_coord = [start_lat,start_long]
end_coord = [end_lat,end_long]
# Create a map centered on the first point in the route
m = folium.Map(location=start_coord, zoom_start=15)
icon1 = folium.Icon(icon='1',prefix='fa')
icon2 = folium.Icon(icon='2',prefix='fa')
for i in range(len(map_coords)):
    if map_coords[i] != start_coord and map_coords[i] != end_coord:
        folium.Marker(location=map_coords[i], icon=folium.Icon(color='green', prefix='fa')).add_to(m)

folium.Marker(location=start_coord, popup='Start',icon=icon1).add_to(m)
folium.Marker(location=end_coord, popup='Finish', icon = icon2).add_to(m)
# Add a polyline to the map using the coordinates
#folium.PolyLine(locations=coordinates, color='red').add_to(m)

m.save('map0.html')
import webbrowser
webbrowser.open('map0.html')

# Print and add the filtered charging stations coordinates as nodes
print('\n Charging stations in the bounding box:')
for i in range(len(filtered_charging_stations)):
    filt_coord = filtered_charging_stations[i]
    graph.add_node(f"Charging Station {i + 1}", pos=filt_coord)
    print(filt_coord)

# Get positions of all nodes for visualization
node_positions = {node: data["pos"] for node, data in graph.nodes(data=True)}

# Draw the graph
plt.figure(figsize=(4, 4))
nx.draw_networkx(graph, pos=node_positions, with_labels=True)
plt.axis("off")
#plt.show()

# Get coordinates from graph nodes
coordinates = []
two_coord=[]
dist_matrix = []
for node, data in graph.nodes(data=True):
    coordinates.append(data["pos"])
print('Working coordinates: ',len(coordinates))
unique_pairs = list(combinations(coordinates, 2))

print(unique_pairs)

# Flatten the nested list and convert it to a set to get unique elements
#unique_elements = filtered_charging_stations
i=0
# Print the unique pairs
for pair in unique_pairs:
    i=i+1
    if i%40 == 0:
        time.sleep(70)
    # Prepare the request body
    dist_matrix.append(list(pair))
    body = {"coordinates": list(pair)}
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': 'YOUR API KEY',
        'Content-Type': 'application/json; charset=utf-8'
    }

    # Make the API request
    call = requests.post('https://api.openrouteservice.org/v2/directions/driving-car', json=body, headers=headers)
    json_data = call.json()
    data = json_data['routes']
    print(data)
    print(data[0]['summary']['distance'])
    dist_matrix.append(data[0]['summary']['distance'])
print('Distance matrix:\n', dist_matrix)

'''#dist_matrix=[[[-123.35825567456034, 48.43176945225633], [-123.3685132381953, 48.42061903292287]], 1954.8,
             [[-123.35825567456034, 48.43176945225633], [-123.36741, 48.421682]], 1969.3, [[-123.35825567456034, 48.43176945225633], [-123.366991, 48.421082]], 1777.2, [[-123.35825567456034, 48.43176945225633], [-123.364496, 48.4220074]], 1455.8, [[-123.35825567456034, 48.43176945225633], [-123.366473, 48.425577]], 1285.0, [[-123.35825567456034, 48.43176945225633], [-123.365639, 48.425067]], 1231.4, [[-123.35825567456034, 48.43176945225633], [-123.363998, 48.42342]], 1260.1, [[-123.35825567456034, 48.43176945225633], [-123.36649, 48.42657]], 1203.9, [[-123.35825567456034, 48.43176945225633], [-123.363773, 48.425103]], 1095.6, [[-123.35825567456034, 48.43176945225633], [-123.365799, 48.428164]], 988.4, [[-123.35825567456034, 48.43176945225633], [-123.366155, 48.42934979]], 835.2, [[-123.35825567456034, 48.43176945225633], [-123.365406, 48.429119]], 780.5, [[-123.35825567456034, 48.43176945225633], [-123.363011, 48.427446]], 772.5, [[-123.35825567456034, 48.43176945225633], [-123.359286344768, 48.4312760361524]], 266.1, [[-123.3685132381953, 48.42061903292287], [-123.36741, 48.421682]], 281.4, [[-123.3685132381953, 48.42061903292287], [-123.366991, 48.421082]], 234.8, [[-123.3685132381953, 48.42061903292287], [-123.364496, 48.4220074]], 502.1, [[-123.3685132381953, 48.42061903292287], [-123.366473, 48.425577]], 853.1, [[-123.3685132381953, 48.42061903292287], [-123.365639, 48.425067]], 703.3, [[-123.3685132381953, 48.42061903292287], [-123.363998, 48.42342]], 720.8, [[-123.3685132381953, 48.42061903292287], [-123.36649, 48.42657]], 968.9, [[-123.3685132381953, 48.42061903292287], [-123.363773, 48.425103]], 854.0, [[-123.3685132381953, 48.42061903292287], [-123.365799, 48.428164]], 1140.1, [[-123.3685132381953, 48.42061903292287], [-123.366155, 48.42934979]], 1320.8, [[-123.3685132381953, 48.42061903292287], [-123.365406, 48.429119]], 1266.1, [[-123.3685132381953, 48.42061903292287], [-123.363011, 48.427446]], 1350.7, [[-123.3685132381953, 48.42061903292287], [-123.359286344768, 48.4312760361524]], 1749.8, [[-123.36741, 48.421682], [-123.366991, 48.421082]], 535.9, [[-123.36741, 48.421682], [-123.364496, 48.4220074]], 442.0, [[-123.36741, 48.421682], [-123.366473, 48.425577]], 540.8, [[-123.36741, 48.421682], [-123.365639, 48.425067]], 590.6, [[-123.36741, 48.421682], [-123.363998, 48.42342]], 608.1, [[-123.36741, 48.421682], [-123.36649, 48.42657]], 856.2, [[-123.36741, 48.421682], [-123.363773, 48.425103]], 741.2, [[-123.36741, 48.421682], [-123.365799, 48.428164]], 1027.4, [[-123.36741, 48.421682], [-123.366155, 48.42934979]], 933.7, [[-123.36741, 48.421682], [-123.365406, 48.429119]], 1153.4, [[-123.36741, 48.421682], [-123.363011, 48.427446]], 1247.7, [[-123.36741, 48.421682], [-123.359286344768, 48.4312760361524]], 1637.1, [[-123.366991, 48.421082], [-123.364496, 48.4220074]], 324.5, [[-123.366991, 48.421082], [-123.366473, 48.425577]], 675.5, [[-123.366991, 48.421082], [-123.365639, 48.425067]], 525.7, [[-123.366991, 48.421082], [-123.363998, 48.42342]], 543.2, [[-123.366991, 48.421082], [-123.36649, 48.42657]], 791.4, [[-123.366991, 48.421082], [-123.363773, 48.425103]], 676.4, [[-123.366991, 48.421082], [-123.365799, 48.428164]], 962.5, [[-123.366991, 48.421082], [-123.366155, 48.42934979]], 1143.3, [[-123.366991, 48.421082], [-123.365406, 48.429119]], 1088.6, [[-123.366991, 48.421082], [-123.363011, 48.427446]], 1173.1, [[-123.366991, 48.421082], [-123.359286344768, 48.4312760361524]], 1572.2, [[-123.364496, 48.4220074], [-123.366473, 48.425577]], 581.6, [[-123.364496, 48.4220074], [-123.365639, 48.425067]], 431.8, [[-123.364496, 48.4220074], [-123.363998, 48.42342]], 449.3, [[-123.364496, 48.4220074], [-123.36649, 48.42657]], 697.4, [[-123.364496, 48.4220074], [-123.363773, 48.425103]], 582.5, [[-123.364496, 48.4220074], [-123.365799, 48.428164]], 868.6, [[-123.364496, 48.4220074], [-123.366155, 48.42934979]], 1049.3, [[-123.364496, 48.4220074], [-123.365406, 48.429119]], 994.6, [[-123.364496, 48.4220074], [-123.363011, 48.427446]], 1079.2, [[-123.364496, 48.4220074], [-123.359286344768, 48.4312760361524]], 1478.3, [[-123.366473, 48.425577], [-123.365639, 48.425067]], 149.8, [[-123.366473, 48.425577], [-123.363998, 48.42342]], 421.4, [[-123.366473, 48.425577], [-123.36649, 48.42657]], 320.1, [[-123.366473, 48.425577], [-123.363773, 48.425103]], 205.2, [[-123.366473, 48.425577], [-123.365799, 48.428164]], 491.3, [[-123.366473, 48.425577], [-123.366155, 48.42934979]], 672.0, [[-123.366473, 48.425577], [-123.365406, 48.429119]], 617.3, [[-123.366473, 48.425577], [-123.363011, 48.427446]], 698.9, [[-123.366473, 48.425577], [-123.359286344768, 48.4312760361524]], 1101.0, [[-123.365639, 48.425067], [-123.363998, 48.42342]], 271.6, [[-123.365639, 48.425067], [-123.36649, 48.42657]], 265.6, [[-123.365639, 48.425067], [-123.363773, 48.425103]], 150.7, [[-123.365639, 48.425067], [-123.365799, 48.428164]], 436.8, [[-123.365639, 48.425067], [-123.366155, 48.42934979]], 617.5, [[-123.365639, 48.425067], [-123.365406, 48.429119]], 562.9, [[-123.365639, 48.425067], [-123.363011, 48.427446]], 644.4, [[-123.365639, 48.425067], [-123.359286344768, 48.4312760361524]], 1046.5, [[-123.363998, 48.42342], [-123.36649, 48.42657]], 537.2, [[-123.363998, 48.42342], [-123.363773, 48.425103]], 380.1, [[-123.363998, 48.42342], [-123.365799, 48.428164]], 708.4, [[-123.363998, 48.42342], [-123.366155, 48.42934979]], 889.1, [[-123.363998, 48.42342], [-123.365406, 48.429119]], 834.5, [[-123.363998, 48.42342], [-123.363011, 48.427446]], 656.0, [[-123.363998, 48.42342], [-123.359286344768, 48.4312760361524]], 1110.8, [[-123.36649, 48.42657], [-123.363773, 48.425103]], 665.2, [[-123.36649, 48.42657], [-123.365799, 48.428164]], 554.6, [[-123.36649, 48.42657], [-123.366155, 48.42934979]], 452.6, [[-123.36649, 48.42657], [-123.365406, 48.429119]], 507.3, [[-123.36649, 48.42657], [-123.363011, 48.427446]], 762.2, [[-123.36649, 48.42657], [-123.359286344768, 48.4312760361524]], 1164.3, [[-123.363773, 48.425103], [-123.365799, 48.428164]], 492.2, [[-123.363773, 48.425103], [-123.366155, 48.42934979]], 672.9, [[-123.363773, 48.425103], [-123.365406, 48.429119]], 618.2, [[-123.363773, 48.425103], [-123.363011, 48.427446]], 491.5, [[-123.363773, 48.425103], [-123.359286344768, 48.4312760361524]], 946.3, [[-123.365799, 48.428164], [-123.366155, 48.42934979]], 267.4, [[-123.365799, 48.428164], [-123.365406, 48.429119]], 322.1, [[-123.365799, 48.428164], [-123.363011, 48.427446]], 588.3, [[-123.365799, 48.428164], [-123.359286344768, 48.4312760361524]], 990.4, [[-123.366155, 48.42934979], [-123.365406, 48.429119]], 54.7, [[-123.366155, 48.42934979], [-123.363011, 48.427446]], 591.4, [[-123.366155, 48.42934979], [-123.359286344768, 48.4312760361524]], 677.6, [[-123.365406, 48.429119], [-123.363011, 48.427446]], 536.7, [[-123.365406, 48.429119], [-123.359286344768, 48.4312760361524]], 622.9,
             [[-123.363011, 48.427446], [-123.359286344768, 48.4312760361524]], 825.7]'''

# Create a set of unique points
unique_points = set(tuple(point) for pair in dist_matrix[::2] for point in pair)
# Create a dictionary to map points to indices
point_indices = {point: index for index, point in enumerate(unique_points)}

# Initialize an empty adjacency matrix
matrix_size = len(unique_points)
adjacency_matrix = [[0] * matrix_size for _ in range(matrix_size)]

# Build the adjacency matrix
for i in range(0, len(dist_matrix), 2):
    start_point = tuple(dist_matrix[i][0])
    end_point = tuple(dist_matrix[i][1])
    distance = dist_matrix[i + 1]

    # Find the indices of the start and end points
    start_index = point_indices[start_point]
    end_index = point_indices[end_point]

    # Update the adjacency matrix with the distance
    adjacency_matrix[start_index][end_index] = distance
    adjacency_matrix[end_index][start_index] = distance

# Print the adjacency matrix
print('Adjacency matrix: ')
for row in adjacency_matrix:
    print(row)
car_range = 1000.0
for row in adjacency_matrix:
    print(row)
    for i in row:
        if i==0:
            pass
        elif i<=car_range:
            flag=0
    if flag==1:
        print("Trip not possible because no viscinity of charging station before ",car_range," meters")
        sys.exit()

for i in range(len(adjacency_matrix[0])):
    for j in range(len(adjacency_matrix)):
        if adjacency_matrix[i][j]==dist_matrix[1]:
            start_ind,end_ind = j,i

# Define the adjacency matrix
adjacency_matrix = np.array(adjacency_matrix)
cities=["_" for i in range(len(adjacency_matrix))]

charge_station=[]
cities[start_ind] = 'Start'
cities[end_ind] = 'End'
for i in range(len(cities)):
    if cities[i]=="_":
        cities[i] = "CS"+str(i+1)
        charge_station.append(cities[i])

# Define other parameters
start_position = cities.index('Start')  # Index of the start position in the adjacency matrix
finish_position = cities.index('End')  # Index of the finish position in the adjacency matrix
charging_stations = [cities.index(cs) for cs in charge_station] # Indices of the charging stations in the adjacency matrix
car_range = 1000.0  # Maximum range of the car in meters
#recharge_rate = 1000.0  # Recharge rate of the car in kilometers per hour

pheromone_matrix = np.ones_like(adjacency_matrix)  # Initialize pheromone levels
pheromone_evaporation = 0.5  # Pheromone evaporation rate
alpha = 2  # Pheromone importance parameter
beta = 1  # Heuristic information importance parameter
num_ants = 200  # Number of ants in the population
num_iterations = 200  # Number of iterations

best_path = None
best_cost = float('inf')

# ACO iteration loop
for iteration in range(num_iterations):
    ant_paths = []

    # Construct solutions for each ant
    for ant in range(num_ants):
        visited = [start_position]  # Start with the start position
        current_range = car_range

        # Move to the next node until reaching the finish position
        while visited[-1] != finish_position:
            current_node = visited[-1]

            # Calculate probabilities for the next node selection
            probabilities = []
            for node in range(len(adjacency_matrix)):
                if node not in visited:
                    distance = adjacency_matrix[current_node, node]
                    if distance <= current_range:
                        pheromone = pheromone_matrix[current_node, node] ** alpha
                        remaining_distance = min(distance, current_range)
                        heuristic = 1 / remaining_distance ** beta
                        probabilities.append(pheromone * heuristic)
                    else:
                        probabilities.append(0)
                else:
                    probabilities.append(0)

            # Check if probabilities are all zero (no valid options)
            if np.sum(probabilities) == 0:
                probabilities = np.ones(len(adjacency_matrix))
                probabilities[visited] = 0

            # Normalize probabilities and select the next node
            probabilities = probabilities / np.sum(probabilities)
            next_node = np.random.choice(range(len(adjacency_matrix)), p=probabilities)

            # Check if recharging is needed at a charging station
            if next_node in charge_station:
                current_range = car_range  # Recharge the car's range to the maximum

            visited.append(next_node)
            current_range -= adjacency_matrix[current_node, next_node]

        ant_paths.append(visited)

    # Update pheromone levels
    pheromone_matrix *= (1 - pheromone_evaporation)  # Evaporation
    for ant_path in ant_paths:
        cost = sum(adjacency_matrix[ant_path[i], ant_path[i+1]] for i in range(len(ant_path)-1))
        if cost < best_cost:
            best_path = ant_path
            best_cost = cost
        for i in range(len(ant_path)-1):
            pheromone_matrix[ant_path[i], ant_path[i+1]] += 1 / cost  # Deposit pheromone
flag = True
# Output the best solution
for i in range(len(best_path)-1):
    if adjacency_matrix[i][i+1] > car_range:
        flag = False
print("Best path:", [cities[i] for i in best_path])
print("Best cost:", best_cost)
'''if flag:
    print("Best path:", [cities[i] for i in best_path])
    print("Best cost:", best_cost)
else:
    print("No feasible path found")
    print("Instead we can reach till {} using path: {} and with a cost of {}".format(cities[best_path[-2]],[cities[i] for i in best_path[:-1]],(best_cost-adjacency_matrix[best_path[-1]][best_path[-2]])))
'''
lst = []
coord = []
for i in range(len(best_path)-1):
    a=(best_path[i],best_path[i+1])
    b = adjacency_matrix[a[0]][a[1]]
    lst.append(b)
#print(lst)

for _ in lst:
    for i in range(len(dist_matrix)):
        if (i % 2) != 0 :
            if _ == dist_matrix[i]:
                coord.append(dist_matrix[i - 1])
        else:
            pass

# Flatten the nested list and convert it to a set to get unique elements
charging_stations = list(set([tuple(item) for sublist in coord for item in sublist]))

body = {"coordinates":charging_stations}
headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
    'Authorization': '5b3ce3597851110001cf62480f7f82686adf4cfda6fea8db623d6c20',
    'Content-Type': 'application/json; charset=utf-8'
}
call = requests.post('https://api.openrouteservice.org/v2/directions/driving-car', json=body, headers=headers)
json_data=call.json()
data = json_data['routes']


import folium
import polyline
for i in range(len(charging_stations)):
    charging_stations[i] = list(charging_stations[i])
map_coords =[]
for i in range(len(charging_stations)):
    a, b = charging_stations[i][0], charging_stations[i][1]
    temp=[]
    temp.append(b)
    temp.append(a)
    map_coords.append(temp)
# Extract the geometry string from the data
geometry = json_data['routes'][0]['geometry']

# Decode the geometry string to get the latitude and longitude coordinates
coordinates = polyline.decode(geometry)

# Create a map centered on the first point in the route
m = folium.Map(location=coordinates[0], zoom_start=15)
start_coord = [start_lat,start_long]
end_coord = [end_lat,end_long]
icon1 = folium.Icon(icon='1',prefix='fa')
icon2 = folium.Icon(icon='2',prefix='fa')
for i in range(len(map_coords)):
    if map_coords[i] != start_coord and map_coords[i] != end_coord:
        folium.Marker(location=map_coords[i], icon=folium.Icon(color='green', prefix='fa')).add_to(m)

folium.Marker(location=start_coord, popup='Start',icon=icon1).add_to(m)
folium.Marker(location=end_coord, popup='Finish', icon = icon2).add_to(m)
# Add a polyline to the map using the coordinates
folium.PolyLine(locations=coordinates, color='red').add_to(m)

m.save('map.html')
import webbrowser
webbrowser.open('map.html')



