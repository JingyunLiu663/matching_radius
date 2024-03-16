import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# G = ox.load_graphml('./input/graph.graphml')
# gdf_nodes, _ = ox.graph_to_gdfs(G)

# # filter out Manhattan area
# manhattan_polygon = ox.geocode_to_gdf('Manhattan, New York, USA')
# nodes_geo = gpd.GeoDataFrame(gdf_nodes, geometry=[Point(xy) for xy in zip(gdf_nodes['x'], gdf_nodes['y'])])
# nodes_in_manhattan = gpd.sjoin(nodes_geo, manhattan_polygon, how="inner", predicate='intersects')

# # Extract the node IDs that are within the Manhattan polygon
# manhattan_node_ids = nodes_in_manhattan.index.tolist()
# # Create a subgraph of G that only contains nodes within Manhattan
# G_manhattan = G.subgraph(manhattan_node_ids)

# ox.save_graphml(G_manhattan, './input/manhattan.graphml')

G = ox.load_graphml('./input/manhattan.graphml')
fig, ax = ox.plot_graph(G, figsize=(12, 12))
fig.savefig("manhattan.png", bbox_inches='tight', dpi=300)