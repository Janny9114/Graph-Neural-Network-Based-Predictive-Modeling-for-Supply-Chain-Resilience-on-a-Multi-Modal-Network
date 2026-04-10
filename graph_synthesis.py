import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, Any, List, Tuple

def set_random_seed(seed: int = 42):
    np.random.seed(seed)

def generate_tier_sizes(
    n_tiers: int = 4,
    base_per_tier: int = 20,
    tier_scaling: List[float] = None
) -> List[int]:
    """
    Generate the number of nodes for each tier.
    Example: more suppliers at upstream tiers, fewer retailers downstream.
    """
    if tier_scaling is None:
        # Example: suppliers > manufacturers > distributors > retailers
        # scaling factors for each tier; multiply by base_per_tier
        tier_scaling = [4.0, 2.5, 1.0, 2.5]

    if len(tier_scaling) != n_tiers:
        raise ValueError("tier_scaling length must equal n_tiers")

    return [int(base_per_tier * s) for s in tier_scaling]

def assign_regions(num_nodes: int, region_names: List[str]) -> List[str]:
    """
    Randomly assign regions to nodes.
    """
    return list(np.random.choice(region_names, size=num_nodes))

def get_country_boundaries():
    """
    Load world country boundaries using geopandas natural earth data.
    Returns a GeoDataFrame with country geometries.
    """
    try:
        # Download and load natural earth low resolution country boundaries directly
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        return world
    except Exception as e:
        print(f"Warning: Could not load natural earth data: {e}")
        return None

def generate_zone_coordinates(regions: List[str], region_names: List[str]) -> Tuple[List[float], List[float]]:
    """
    Generate realistic geographic coordinates for nodes based on their country/region.
    Uses geopandas to get actual country boundaries and places nodes within them.
    
    Args:
        regions: List of region assignments for each node
        region_names: List of all possible region names
        
    Returns:
        Tuple of (longitude_coordinates, latitude_coordinates)
    """
    # Load world boundaries
    world = get_country_boundaries()
    
    # Map region names to country names in natural earth dataset
    region_to_country = {
        "China 1": "China",
        "China 2": "China",
        "Mexico": "Mexico",
        "Netherlands": "Netherlands",
        "England": "United Kingdom",
        "United States 1": "United States of America",
        "Norway": "Norway",
        "India": "India",
        "Kenya": "Kenya"
    }
    
    # Generate coordinates for each node
    lon_coords = []
    lat_coords = []
    
    if world is not None:
        # Check which column name is used for country names
        name_column = None
        for col in ['name', 'NAME', 'ADMIN', 'NAME_EN', 'SOVEREIGNT']:
            if col in world.columns:
                name_column = col
                break
        
        if name_column is None:
            print("Could not find country name column, using fallback...")
            world = None
        
        for region in regions:
            if world is None:
                break
                
            country_name = region_to_country.get(region, region)
            
            # Get the country geometry
            country_geom = world[world[name_column] == country_name]
            
            if not country_geom.empty:
                # Get the country's bounding box
                bounds = country_geom.total_bounds  # [minx, miny, maxx, maxy]
                geometry = country_geom.geometry.iloc[0]
                
                # Generate random points within the country boundary
                max_attempts = 100
                for attempt in range(max_attempts):
                    # Random point within bounding box
                    lon = np.random.uniform(bounds[0], bounds[2])
                    lat = np.random.uniform(bounds[1], bounds[3])
                    point = Point(lon, lat)
                    
                    # Check if point is within country boundary
                    if geometry.contains(point):
                        lon_coords.append(lon)
                        lat_coords.append(lat)
                        break
                else:
                    # If we couldn't find a point inside, use centroid
                    centroid = geometry.centroid
                    lon_coords.append(centroid.x)
                    lat_coords.append(centroid.y)
            else:
                # Fallback: use approximate coordinates if country not found
                fallback_coords = {
                    "China": (105.0, 35.0),
                    "Norway": (10.0, 60.0),
                    "Mexico": (-102.0, 23.0),
                    "Netherlands": (5.3, 52.1),
                    "United Kingdom": (-1.5, 52.5),
                    "United States of America": (-95.0, 37.0),
                    "India": (78.0, 22.0),
                    "Kenya": (37.9, -0.0)
                }
                lon, lat = fallback_coords.get(country_name, (0.0, 0.0))
                # Add some random offset
                lon += np.random.uniform(-5, 5)
                lat += np.random.uniform(-3, 3)
                lon_coords.append(lon)
                lat_coords.append(lat)
    else:
        # Fallback if geopandas data not available
        print("Using fallback coordinate generation...")
        fallback_coords = {
            "China 1": (105.0, 35.0),
            "China 2": (115.0, 30.0),
            "Mexico": (-102.0, 23.0),
            "Netherlands": (5.3, 52.1),
            "England": (-1.5, 52.5),
            "United States 1": (-95.0, 37.0),
            "Norway": (10.0, 60.0),
            "India": (78.0, 22.0),
            "Kenya": (37.9, -0.0)
        }
        
        for region in regions:
            lon, lat = fallback_coords.get(region, (0.0, 0.0))
            # Add random offset within ~500km
            lon += np.random.uniform(-5, 5)
            lat += np.random.uniform(-3, 3)
            lon_coords.append(lon)
            lat_coords.append(lat)
    
    return lon_coords, lat_coords

def generate_node_features(
    tier_index: int,
    num_nodes: int,
    region_names: List[str]
) -> pd.DataFrame:
    regions = assign_regions(num_nodes, region_names)

    base_capacity = {
        0: (1000, 200),
        1: (700, 150),
        2: (500, 100),
        3: (200, 80)
    }
    base_risk = {
        0: (0.2, 0.08),
        1: (0.25, 0.10),
        2: (0.3, 0.12),
        3: (0.25, 0.09)
    }
    # Realistic cost factors by tier
    # Tier 0: Suppliers (0.60-0.85) - raw materials, lower baseline value
    # Tier 1: Manufacturers (0.85-1.10) - value addition, economies of scale
    # Tier 2: Distributors (1.10-1.40) - finished goods, higher capital tied up
    # Tier 3: Retailers (1.40-2.00) - prime real estate, customer-facing costs
    base_cost_factor = {
        0: (0.725, 0.0625),  # mean=0.725, std to cover 0.60-0.85 range (~95% within 2 std)
        1: (0.975, 0.0625),  # mean=0.975, std to cover 0.85-1.10 range
        2: (1.25, 0.075),    # mean=1.25, std to cover 1.10-1.40 range
        3: (1.70, 0.15)      # mean=1.70, std to cover 1.40-2.00 range
    }

    cap_mean, cap_std = base_capacity.get(tier_index, (400, 120))
    risk_mean, risk_std = base_risk.get(tier_index, (0.25, 0.10))
    cost_mean, cost_std = base_cost_factor.get(tier_index, (1.0, 0.2))

    capacity = np.maximum(
        np.random.normal(loc=cap_mean, scale=cap_std, size=num_nodes),
        10
    )

    # Generate cost_factor with tier-specific clipping ranges
    cost_factor_raw = np.random.normal(loc=cost_mean, scale=cost_std, size=num_nodes)
    
    # Tier-specific clipping ranges
    clip_ranges = {
        0: (0.60, 0.85),
        1: (0.85, 1.10),
        2: (1.10, 1.40),
        3: (1.40, 2.00)
    }
    clip_min, clip_max = clip_ranges.get(tier_index, (0.5, 2.5))
    
    cost_factor = np.round(
        np.clip(cost_factor_raw, clip_min, clip_max),
        2  # Round to 2 decimal places
    )
    risk_level = np.clip(
        np.random.normal(loc=risk_mean, scale=risk_std, size=num_nodes),
        0.01, 0.99
    )
    
    # Generate reliability score using Beta distribution skewed towards high reliability
    # Research basis:
    # - Chopra & Sodhi (2004): "Managing Risk to Avoid Supply-Chain Breakdown" - larger firms have better reliability
    # - Christopher & Peck (2004): "Building the Resilient Supply Chain" - capacity correlates with reliability
    # - Sheffi & Rice (2005): "A Supply Chain View of the Resilient Enterprise" - risk inversely affects reliability
    
    # Base reliability from Beta distribution (skewed towards 0.95)
    # Beta(α=20, β=2) gives mean ≈ 0.91, heavily skewed towards high values
    base_reliability = np.random.beta(a=20, b=2, size=num_nodes)
    
    # Normalize capacity to [0, 1] for correlation calculation
    capacity_normalized = (capacity - capacity.min()) / (capacity.max() - capacity.min() + 1e-8)
    
    # Apply correlations:
    # 1. Positive correlation with capacity (+0.05 * normalized_capacity)
    #    Larger nodes are more reliable (economies of scale, better processes)
    # 2. Negative correlation with risk (-0.15 * risk_level)
    #    Higher risk reduces reliability
    reliability = base_reliability + 0.05 * capacity_normalized - 0.15 * risk_level
    
    # Clip to realistic range [0.60, 0.99]
    # Most supply chain nodes are reasonably reliable (60%+), but not perfect
    reliability = np.clip(reliability, 0.60, 0.99)

    # Generate x and y coordinates based on zone/region
    x_coords, y_coords = generate_zone_coordinates(regions, region_names)

    df = pd.DataFrame({
        "tier": tier_index,
        "region": regions,
        "x": x_coords,
        "y": y_coords,
        "capacity": capacity,
        "cost_factor": cost_factor,
        "risk_level": risk_level,
        "reliability": reliability
    })

    return df


def connect_tiers(
    upstream_nodes: List[int],
    downstream_nodes: List[int],
    tier_connection: str = "generic",
    connection_bias: str = "random"
) -> List[Tuple[int, int]]:
    """
    Create directed edges from upstream tier to downstream tier with realistic degree constraints.
    
    Realistic degree constraints:
    - Suppliers -> Manufacturers: Each Manufacturer selects 2-4 Suppliers (multi-sourcing)
    - Manufacturers -> Distributors: Each Distributor selects 1-3 Manufacturers
    - Distributors -> Retailers: Each Retailer connects to 1 primary Distributor (hub-and-spoke)
                                 with 10% chance of 2nd backup Distributor
    
    Args:
        upstream_nodes: List of upstream node IDs
        downstream_nodes: List of downstream node IDs
        tier_connection: Type of connection ("supplier_to_mfg", "mfg_to_dist", "dist_to_retail", "generic")
        connection_bias: "random" or "capacity_preferential" (placeholder)
    
    Returns:
        List of edges (source, target)
    """
    edges = []

    if len(upstream_nodes) == 0 or len(downstream_nodes) == 0:
        return edges

    # Apply realistic degree constraints based on tier connection
    for dn in downstream_nodes:
        if tier_connection == "supplier_to_mfg":
            # Manufacturers: 2-4 suppliers (multi-sourcing)
            n_connections = np.random.randint(2, 5)
            n_connections = min(n_connections, len(upstream_nodes))
            
        elif tier_connection == "mfg_to_dist":
            # Distributors: 1-3 manufacturers
            n_connections = np.random.randint(1, 4)
            n_connections = min(n_connections, len(upstream_nodes))
            
        elif tier_connection == "dist_to_retail":
            # Retailers: 1 primary distributor + 10% chance of backup
            n_connections = 1
            if np.random.random() < 0.10:  # 10% chance of backup
                n_connections = 2
            n_connections = min(n_connections, len(upstream_nodes))
            
        else:
            # Generic: Poisson distribution (fallback for other connections)
            n_connections = np.random.poisson(lam=2.5)
            n_connections = np.clip(n_connections, 1, len(upstream_nodes))
        
        # Pick distinct upstream nodes
        selected_upstream = np.random.choice(upstream_nodes, size=n_connections, replace=False)
        for u in selected_upstream:
            edges.append((u, dn))

    return edges


def generate_edge_features(
    G: nx.DiGraph,
    node_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate edge-level features including hierarchical flow quantities.
    
    Flow ranges by tier connection:
    - Suppliers (Tier 0) -> Manufacturers (Tier 1): 10,000 to 50,000 units
    - Manufacturers (Tier 1) -> Distributors (Tier 2): 5,000 to 20,000 units
    - Distributors (Tier 2) -> Retailers (Tier 3): 1,000 to 5,000 units
    """
    edges = list(G.edges())
    n_edges = len(edges)

    lead_times = []
    transport_costs = []
    capacity_shares = []
    disruption_probs = []
    flow_quantities = []

    for u, v in edges:
        region_u = node_df.loc[u, "region"]
        region_v = node_df.loc[v, "region"]
        tier_u = node_df.loc[u, "tier"]
        tier_v = node_df.loc[v, "tier"]

        # Hierarchical flow quantities based on tier connection
        if tier_u == 0 and tier_v == 1:
            # Suppliers -> Manufacturers: 10,000 to 50,000 units
            flow = np.random.uniform(10000, 50000)
        elif tier_u == 1 and tier_v == 2:
            # Manufacturers -> Distributors: 5,000 to 20,000 units
            flow = np.random.uniform(5000, 20000)
        elif tier_u == 2 and tier_v == 3:
            # Distributors -> Retailers: 1,000 to 5,000 units
            flow = np.random.uniform(1000, 5000)
        else:
            # Fallback for any other connections
            flow = np.random.uniform(1000, 10000)

        # Domestic vs international
        if region_u == region_v:
            lt = np.random.normal(loc=3, scale=1.0)   # short lead time
            cost = np.random.normal(loc=100, scale=20)
        else:
            lt = np.random.normal(loc=10, scale=3.0)  # longer lead time
            cost = np.random.normal(loc=300, scale=50)

        lt = max(1.0, lt)
        cost = max(20.0, cost)

        # Capacity share: fraction of upstream node's capacity sent to this downstream
        share = np.clip(np.random.beta(a=2, b=5), 0.01, 0.7)

        # Disruption probability: combine node risks + random noise
        risk_u = node_df.loc[u, "risk_level"]
        risk_v = node_df.loc[v, "risk_level"]
        d_prob = np.clip(
            0.5 * risk_u + 0.3 * risk_v + np.random.normal(0, 0.05),
            0.01, 0.95
        )

        lead_times.append(lt)
        transport_costs.append(cost)
        capacity_shares.append(share)
        disruption_probs.append(d_prob)
        flow_quantities.append(flow)

    edge_df = pd.DataFrame({
        "source": [u for u, v in edges],
        "target": [v for u, v in edges],
        "lead_time": lead_times,
        "transport_cost": transport_costs,
        "capacity_share": capacity_shares,
        "disruption_probability": disruption_probs,
        "flow_quantity": flow_quantities
    })

    return edge_df


def build_synthetic_supply_chain_graph(
    n_tiers: int = 4,
    base_per_tier: int = 20,
    tier_scaling: List[float] = None,
    region_names: List[str] = None,
    avg_degree_between_tiers: float = 2.5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Master function:
    - Creates a multi-tier directed graph
    - Generates node and edge attributes
    - Returns networkx graph + node/edge DataFrames
    """
    set_random_seed(seed)

    
    if region_names is None:
        region_names = ["China 1", "China 2", "Mexico", "Netherlands", "England", "United States 1", "Norway", "India", "Kenya"]

    tier_sizes = generate_tier_sizes(
        n_tiers=n_tiers,
        base_per_tier=base_per_tier,
        tier_scaling=tier_scaling
    )

    # Build node features tier by tier
    node_dfs = []
    node_ids = []
    current_id = 0

    for t, size in enumerate(tier_sizes):
        df_t = generate_node_features(
            tier_index=t,
            num_nodes=size,
            region_names=region_names
        )
        df_t.index = range(current_id, current_id + size)
        node_dfs.append(df_t)
        node_ids.append(list(df_t.index))
        current_id += size

    node_df = pd.concat(node_dfs, axis=0)
    node_df.index.name = "node_id"

    # Create edges between consecutive tiers with realistic degree constraints
    edges = []
    tier_connection_types = ["supplier_to_mfg", "mfg_to_dist", "dist_to_retail"]
    
    for t in range(n_tiers - 1):
        up_nodes = node_ids[t]
        down_nodes = node_ids[t + 1]
        
        # Determine connection type
        if t < len(tier_connection_types):
            connection_type = tier_connection_types[t]
        else:
            connection_type = "generic"
        
        edges_t = connect_tiers(
            upstream_nodes=up_nodes,
            downstream_nodes=down_nodes,
            tier_connection=connection_type
        )
        edges.extend(edges_t)

    # Create networkx DiGraph and add attributes
    G = nx.DiGraph()
    for nid, row in node_df.iterrows():
        G.add_node(
            nid,
            tier=int(row["tier"]),
            region=row["region"],
            x=float(row["x"]),
            y=float(row["y"]),
            capacity=float(row["capacity"]),
            cost_factor=float(row["cost_factor"]),
            risk_level=float(row["risk_level"])
        )

    G.add_edges_from(edges)

    # Create edge attribute DataFrame and add to graph
    edge_df = generate_edge_features(G, node_df)
    for _, row in edge_df.iterrows():
        u = int(row["source"])
        v = int(row["target"])
        G[u][v]["lead_time"] = float(row["lead_time"])
        G[u][v]["transport_cost"] = float(row["transport_cost"])
        G[u][v]["capacity_share"] = float(row["capacity_share"])
        G[u][v]["disruption_probability"] = float(row["disruption_probability"])

    return {
        "graph": G,
        "node_df": node_df,
        "edge_df": edge_df,
        "tier_sizes": tier_sizes
    }

        

def export_to_csv(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    node_path: str = "nodes.csv",
    edge_path: str = "edges.csv"
):
    """
    Save nodes and edges to CSV for later use in GNN frameworks.
    """
    node_df.reset_index().to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)
    print(f"Saved nodes to {node_path} and edges to {edge_path}")


if __name__ == "__main__":
    # Read the CSV file with proper header handling
    region_risk_df = pd.read_csv("ResilienceIndexRegions.csv")
    
    # The CSV has column names in the first row, pandas will read them automatically
    country_score = dict(zip(region_risk_df["Country"], region_risk_df["Country Score"].astype(float)))

    data = build_synthetic_supply_chain_graph(
        n_tiers=4,
        base_per_tier=20,  # Changed from 20 to 500 for 5000 nodes
        tier_scaling=[3.0, 2.5, 2.0, 2.5],
        avg_degree_between_tiers=3.0,
        seed=123
    )

    G = data["graph"]
    node_df = data["node_df"]
    edge_df = data["edge_df"]

    print("Tier sizes:", data["tier_sizes"])
    print("Total nodes:", G.number_of_nodes())
    print("Total edges:", G.number_of_edges())

    print("\nSample nodes:")
    print(node_df.head())

    print("\nSample edges:")
    print(edge_df.head())

    export_to_csv(node_df, edge_df,
                  node_path="synthetic_nodes.csv",
                  edge_path="synthetic_edges.csv")