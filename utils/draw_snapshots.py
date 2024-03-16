import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from config import *
from utilities import *
from simulator_env import Simulator
import networkx as nx
import osmnx as ox
import shapely.wkt
from shapely.geometry import LineString
import matplotlib.patches as patches

'''
The subsystem of the control center that visualizes the results
'''

def simplify_graph_geometry(G):
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            # Convert the WKT string to a shapely LineString object
            line = shapely.wkt.loads(data['geometry'])
            if isinstance(line, LineString):
                # Simplify the LineString to just the start and end points
                data['geometry'] = LineString([line.coords[0], line.coords[-1]])

    return G

def drawRoadNetwork(graphml_file):
    # Load the graph from the GraphML file
    G = nx.read_graphml(graphml_file)
    # Simplify the geometry data in the graph
    G = simplify_graph_geometry(G)
    # Draw the graph using OSMnx
    fig, ax = ox.plot_graph(G, edge_color='grey', 
                            show=False, close=False)
    return ax

def drawVehicles(ax, driver_table, dot_size=10):
    # status：0 cruise (park or cruise)， 1 delivery，2 pickup, 3 offline, 4 reposition
    colors = {0: 'green', 1: 'blue', 2: 'purple', 3: 'red', 4:'dark_green'}
    for _, driver in driver_table.iterrows():
        color = colors[driver.status]
        ax.scatter(driver.lng, driver.lat, color=color, s=dot_size)
    return ax

def drawOrders(ax, order_table, dot_size=10):
    for _, order in order_table.iterrows():
        ax.scatter(order.origin_lng, order.origin_lat, color="orange", s=dot_size)
    return ax

def drawDestinations(ax, order_table, dot_size=10):
    for _, order in order_table.iterrows():
        ax.scatter(order.dest_lng, order.dest_lat, s=dot_size)
    return ax

def drawVehiclesByGrid(ax, driver_table, dot_size=10):
    # Map each unique grid_id to a color
    grid_ids = driver_table['grid_id'].unique()
    color_map = plt.get_cmap('rainbow', len(grid_ids))
    # Create a dictionary that maps grid_id to color
    colors = {grid_id: color_map(i) for i, grid_id in enumerate(grid_ids)}
    for _, driver in driver_table.iterrows():
        color = colors[driver.grid_id]
        ax.scatter(driver.lng, driver.lat, color=color, s=dot_size)
    return ax

def draw_simulation(simulator, graphml_file, experiment_log_name, epoch, time_step):
    # Draw the road network and vehicles
    fig, ax = plt.subplots() 
    ax = drawRoadNetwork(graphml_file)
    drawVehicles(ax, simulator.driver_table)
    drawOrders(ax, simulator.wait_requests)
    # Add a title to the plot
    plt.title(f'Simulation at Time step: {time_step}')
    # Add a text annotation for the timestep
    plt.text(0.01, 0.01, f'Time step: {time_step}', transform=ax.transAxes)
    # create a folder for each experiment
    os.makedirs(f"draw_snapshots/{experiment_log_name}", exist_ok=True)
    parameter_path = os.path.join("draw_snapshots", f'{experiment_log_name}/epoch_{epoch}_step{time_step}.png')
    plt.savefig(parameter_path)
    # Close the figure to free up memory
    plt.close(fig)

if __name__ == "__main__":
    # test
    # initiliza the simulator
    simulator = Simulator(**env_params)
    simulator.experiment_date = "2015-05-04"
    simulator.reset()
    # draw_simulation(simulator, "./input/graph.graphml", 0, 0)
    fig, ax = plt.subplots() 
    ax = drawRoadNetwork("./input/graph.graphml")
    with open("input/drivers_500.pickle", "rb") as f:
        drivers = pickle.load(f)
    drawVehiclesByGrid(ax, drivers)
    plt.savefig(f'drivers_by_grid.png')
    # Close the figure to free up memory
    plt.close(fig)
