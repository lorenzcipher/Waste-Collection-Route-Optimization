import streamlit as st
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import geopy.distance
from collections import defaultdict
import time
from functools import lru_cache
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import os

# ================== 1. DATA LOADING AND PREPROCESSING ==================

def load_and_preprocess_data(filename):
    """Optimized data loading and preprocessing function."""
    dtype_spec = {
        'numero': str,
        'type': str,
        'longitude': str,
        'latitude': str,
        'volume': str,
        'etat': str,
        'population': str,
        'stock_initial_t': str,
        'seuil_critique_t': str,
        'frequence_generation_t': str,
        'capacite_bac': str,
        'nb_rotations_jour': str
    }
    
    if isinstance(filename, str):
        df = pd.read_csv(filename, dtype=dtype_spec)
    else:  # file upload object
        df = pd.read_csv(filename, dtype=dtype_spec)
    
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    numeric_cols = [
        "longitude", "latitude", "volume", "etat", "population",
        "stock_initial_t", "seuil_critique_t", "frequence_generation_t",
        "capacite_bac", "nb_rotations_jour"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
                .replace('', np.nan)
                .astype(float)
                .fillna(0)
            )
    
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.replace(".0", "", regex=False)
    
    return df

# ================== 2. ENTITY CLASSES ==================

class Camion:
    __slots__ = ['numero', 'volume_m3', 'capacite_t']
    
    def __init__(self, numero, volume_m3):
        self.numero = str(numero)
        self.volume_m3 = float(volume_m3)
        self.capacite_t = self.volume_m3 * 0.38
        
    def __repr__(self):
        return f"Camion(num={self.numero}, vol={self.volume_m3}m³, cap={self.capacite_t:.1f}t)"

class PointCollecte:
    __slots__ = [
        'numero', 'longitude', 'latitude', 'population', 
        'stock_initial_t', 'seuil_critique_t', 'frequence_generation_t',
        'capacite_bac', 'nb_rotations_jour'
    ]
    
    def __init__(self, numero, longitude, latitude, population, 
                 stock_initial_t, seuil_critique_t, frequence_generation_t,
                 capacite_bac, nb_rotations_jour):
        self.numero = str(numero)
        self.longitude = float(longitude)
        self.latitude = float(latitude)
        self.population = int(float(population))
        self.stock_initial_t = float(stock_initial_t)
        self.seuil_critique_t = float(seuil_critique_t)
        self.frequence_generation_t = float(frequence_generation_t)
        self.capacite_bac = float(capacite_bac)
        self.nb_rotations_jour = int(float(nb_rotations_jour))
        
    def __repr__(self):
        return f"Point({self.numero}, stock={self.stock_initial_t:.1f}t, pop={self.population})"

class Location:
    __slots__ = ['longitude', 'latitude']
    
    def __init__(self, longitude, latitude):
        self.longitude = float(longitude)
        self.latitude = float(latitude)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(lon={self.longitude:.6f}, lat={self.latitude:.6f})"

class Parc(Location):
    pass

class CET(Location):
    pass

# ================== 3. DATA EXTRACTION ==================

def extract_entities(df):
    camions = [
        Camion(row['numero'], row['volume'])
        for _, row in df[df['type'] == "1"].iterrows()
    ]
    
    parc_row = df[df['type'] == "2"].iloc[0]
    parc = Parc(parc_row['longitude'], parc_row['latitude'])
    
    cet_row = df[df['type'] == "4"].iloc[0]
    cet = CET(cet_row['longitude'], cet_row['latitude'])
    
    points_collecte = [
        PointCollecte(
            row['numero'], row['longitude'], row['latitude'],
            row['population'], row['stock_initial_t'],
            row['seuil_critique_t'], row['frequence_generation_t'],
            row['capacite_bac'], row.get('nb_rotations_jour', 1)
        )
        for _, row in df[df['type'] == "3"].iterrows()
    ]
    
    camions_dict = {c.numero: c for c in camions}
    points_dict = {pc.numero: pc for pc in points_collecte}
    
    return camions, parc, cet, points_collecte, camions_dict, points_dict

# ================== 4. DISTANCE MATRIX ==================

@lru_cache(maxsize=100000)
def cached_distance(lat1, lon1, lat2, lon2):
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).km

distance_cache = {}

def precompute_distances(points, parc, cet):
    all_locations = [parc] + points + [cet]
    n = len(all_locations)
    for i in range(n):
        for j in range(i+1, n):
            loc1 = all_locations[i]
            loc2 = all_locations[j]
            key = (i, j)
            distance_cache[key] = cached_distance(
                loc1.latitude, loc1.longitude, 
                loc2.latitude, loc2.longitude
            )

def get_distance(idx1, idx2):
    if idx1 == idx2:
        return 0.0
    key = (min(idx1, idx2), max(idx1, idx2))
    return distance_cache[key]

def calculate_route_distance_fast(route_points, point_list, parc, cet):
    if not route_points:
        return 0.0
    
    total = get_distance(0, route_points[0])  # depot to first
    
    for i in range(len(route_points)-1):
        total += get_distance(route_points[i], route_points[i+1])
    
    total += get_distance(route_points[-1], len(point_list)+1)  # to landfill
    total += 24.2  # landfill to depot
    
    return total

# ================== 5. SOLUTION GENERATION ==================

def generate_sweep_solution(points, trucks, parc):
    point_angles = [
        math.atan2(p.latitude - parc.latitude, p.longitude - parc.longitude)
        for p in points
    ]
    sorted_indices = sorted(range(len(points)), key=lambda i: point_angles[i])
    sorted_indices = [i+1 for i in sorted_indices]  # 1-based
    
    solution = []
    truck_idx = 0
    remaining_capacity = trucks[truck_idx % len(trucks)].capacite_t
    current_route = []
    current_load = 0
    
    for idx in sorted_indices:
        point = points[idx-1]
        if current_load + point.stock_initial_t <= remaining_capacity + 1e-6:
            current_route.append(idx)
            current_load += point.stock_initial_t
        else:
            if current_route:
                solution.append({
                    "truck": trucks[truck_idx % len(trucks)].numero,
                    "points": current_route.copy(),
                    "load": current_load
                })
            truck_idx += 1
            current_route = [idx]
            current_load = point.stock_initial_t
            remaining_capacity = trucks[truck_idx % len(trucks)].capacite_t
            
    if current_route:
        solution.append({
            "truck": trucks[truck_idx % len(trucks)].numero,
            "points": current_route.copy(),
            "load": current_load
        })
        
    return solution

def generate_nearest_neighbor_solution(points, trucks):
    unvisited = list(range(1, len(points)+1))  # 1-based indices
    solution = []
    truck_idx = 0
    current_route = []
    current_load = 0
    remaining_capacity = trucks[truck_idx % len(trucks)].capacite_t
    current_point = 0  # depot index
    
    while unvisited:
        nearest = None
        min_dist = float('inf')
        
        for p in unvisited:
            point = points[p-1]
            dist = get_distance(current_point, p)
            if (dist < min_dist and 
                current_load + point.stock_initial_t <= remaining_capacity + 1e-6):
                nearest = p
                min_dist = dist
        
        if nearest is not None:
            current_route.append(nearest)
            current_load += points[nearest-1].stock_initial_t
            unvisited.remove(nearest)
            current_point = nearest
        else:
            if current_route:
                solution.append({
                    "truck": trucks[truck_idx % len(trucks)].numero,
                    "points": current_route.copy(),
                    "load": current_load
                })
            truck_idx += 1
            current_route = []
            current_load = 0
            remaining_capacity = trucks[truck_idx % len(trucks)].capacite_t
            current_point = 0
    
    if current_route:
        solution.append({
            "truck": trucks[truck_idx % len(trucks)].numero,
            "points": current_route.copy(),
            "load": current_load
        })
        
    return solution

def generate_random_solution(points, trucks):
    point_indices = list(range(1, len(points)+1))
    random.shuffle(point_indices)
    solution = []
    truck_idx = 0
    remaining_capacity = trucks[truck_idx % len(trucks)].capacite_t
    current_route = []
    current_load = 0
    
    for idx in point_indices:
        point = points[idx-1]
        if current_load + point.stock_initial_t <= remaining_capacity + 1e-6:
            current_route.append(idx)
            current_load += point.stock_initial_t
        else:
            if current_route:
                solution.append({
                    "truck": trucks[truck_idx % len(trucks)].numero,
                    "points": current_route.copy(),
                    "load": current_load
                })
            truck_idx += 1
            current_route = [idx]
            current_load = point.stock_initial_t
            remaining_capacity = trucks[truck_idx % len(trucks)].capacite_t
            
    if current_route:
        solution.append({
            "truck": trucks[truck_idx % len(trucks)].numero,
            "points": current_route.copy(),
            "load": current_load
        })
        
    return solution

def optimized_initial_population(points, trucks, parc, size=20):
    population = []
    # Convert trucks to list if it's a dict_values object
    trucks_list = list(trucks) if not isinstance(trucks, list) else trucks
    population.append(generate_sweep_solution(points, trucks_list, parc))
    population.append(generate_nearest_neighbor_solution(points, trucks_list))
    for _ in range(size - 2):
        population.append(generate_random_solution(points, trucks_list))
    return population

# ================== 6. GENETIC ALGORITHM CORE ==================

fitness_params = {
    "capacity_penalty": 1e6,
    "critical_point_penalty": 1e7,
    "distance_weight": 2.0,
    "route_count_weight": 200,
    "distance_penalty_threshold": 350,
    "distance_penalty_factor": 1e4
}

def fast_fitness_function(solution, stocks, critical_points, params, point_list, parc, cet, camions_dict):
    total_distance = 0.0
    penalty = 0.0
    served_points = set()
    
    for route in solution:
        truck_capacity = camions_dict[route["truck"]].capacite_t
        route_dist = calculate_route_distance_fast(route["points"], point_list, parc, cet)
        total_distance += route_dist
        route_load = sum(stocks[p] for p in route["points"])
        served_points.update(route["points"])
        
        if route_load > truck_capacity + 1e-6:
            penalty += params["capacity_penalty"] * (route_load - truck_capacity)
    
    unserved_critical = len(set(critical_points) - served_points)
    if unserved_critical > 0:
        penalty += params["critical_point_penalty"] * unserved_critical
    
    if total_distance > params["distance_penalty_threshold"]:
        penalty += params["distance_penalty_factor"] * (total_distance - params["distance_penalty_threshold"])
    
    fitness = (params["distance_weight"] * total_distance + 
               params["route_count_weight"] * len(solution) + 
               penalty)
    
    return fitness, total_distance

def evaluate_population_sequential(population, current_stocks, critical_points, params, point_list, parc, cet, camions_dict):
    fitness_values = []
    distances = []
    for solution in population:
        fitness, distance = fast_fitness_function(solution, current_stocks, critical_points, params, point_list, parc, cet, camions_dict)
        fitness_values.append(fitness)
        distances.append(distance)
    return fitness_values, distances

def tournament_selection(population, fitness_values, tournament_size=3):
    selected = []
    n = len(population)
    for _ in range(n):
        tournament_indices = random.sample(range(n), min(tournament_size, n))
        winner_idx = min(tournament_indices, key=lambda i: fitness_values[i])
        selected.append(copy.deepcopy(population[winner_idx]))
    return selected

def simple_crossover(parent1, parent2, current_stocks, camions, point_list, camions_dict):
    if len(parent1) <= 1 or len(parent2) <= 1:
        return random.choice([parent1, parent2])
    cut_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child = parent1[:cut_point] + parent2[cut_point:]
    return repair_solution_fast(child, current_stocks, camions, point_list, camions_dict)

def repair_solution_fast(solution, current_stocks, camions, point_list, camions_dict):
    if not solution:
        return solution
    
    served_count = defaultdict(int)
    for route in solution:
        for point in route["points"]:
            served_count[point] += 1
    
    duplicates = {p for p, count in served_count.items() if count > 1}
    if duplicates:
        for route in solution:
            original_points = route["points"][:]
            route["points"] = []
            for p in original_points:
                if p in duplicates:
                    duplicates.discard(p)
                    route["points"].append(p)
                elif p not in duplicates:
                    route["points"].append(p)
        solution = [route for route in solution if route["points"]]
    
    all_points = set(range(1, len(point_list) + 1))
    served_points = set(served_count.keys())
    missing_points = all_points - served_points
    
    for point in missing_points:
        added = False
        point_stock = current_stocks.get(point, 0)
        for route in solution:
            truck = camions_dict[route["truck"]]
            current_load = sum(current_stocks.get(p, 0) for p in route["points"])
            if current_load + point_stock <= truck.capacite_t + 1e-6:
                route["points"].append(point)
                added = True
                break
        if not added:
            truck = random.choice(camions)
            solution.append({
                "truck": truck.numero,
                "points": [point],
                "load": point_stock
            })
    return solution

def fast_mutation(solution, mutation_rate=0.1):
    if random.random() > mutation_rate:
        return solution
    mutated = copy.deepcopy(solution)
    mutation_type = random.choice(['swap', 'reverse'])
    if mutation_type == 'swap' and len(mutated) >= 2:
        route1, route2 = random.sample(mutated, 2)
        if route1["points"] and route2["points"]:
            p1 = random.choice(route1["points"])
            p2 = random.choice(route2["points"])
            route1["points"][route1["points"].index(p1)] = p2
            route2["points"][route2["points"].index(p2)] = p1
    elif mutation_type == 'reverse':
        route = random.choice(mutated)
        if len(route["points"]) >= 2:
            i, j = sorted(random.sample(range(len(route["points"])), 2))
            route["points"][i:j+1] = route["points"][i:j+1][::-1]
    return mutated

def run_streamlit_genetic_algorithm(generations, population_size, crossover_rate, mutation_rate, point_list, parc, cet, camions_dict):
    population = optimized_initial_population(point_list, camions_dict.values(), parc, population_size)
    best_solution = None
    best_fitness = float('inf')
    best_distance = float('inf')
    fitness_history = []
    distance_history = []
    no_improvement_count = 0
    
    current_stocks = {i: p.stock_initial_t for i, p in enumerate(point_list, 1)}
    thresholds = {i: p.seuil_critique_t for i, p in enumerate(point_list, 1)}
    critical_points = get_critical_points(point_list, current_stocks, thresholds)
    
    for generation in range(generations):
        fitness_values, distances = evaluate_population_sequential(
            population, current_stocks, critical_points, fitness_params,
            point_list, parc, cet, camions_dict
        )
        
        current_best_fitness = min(fitness_values)
        best_idx = fitness_values.index(current_best_fitness)
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = copy.deepcopy(population[best_idx])
            best_distance = distances[best_idx]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        fitness_history.append(current_best_fitness)
        distance_history.append(min(distances))
        
        if no_improvement_count >= 8 or best_distance <= 300:
            break
        
        selected = tournament_selection(population, fitness_values)
        new_population = []
        elite_size = max(1, population_size // 5)
        elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:elite_size]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(population[idx]))
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            if random.random() < crossover_rate:
                child = simple_crossover(
                    parent1, parent2, current_stocks,
                    list(camions_dict.values()), point_list, camions_dict
                )
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
            child = fast_mutation(child, mutation_rate)
            new_population.append(child)
        
        population = new_population[:population_size]
    
    return best_solution, fitness_history, distance_history

def get_critical_points(points, stocks, thresholds):
    return [i for i, point in enumerate(points, 1) if stocks[i] >= thresholds[i] - 1e-6]

# ================== 7. STREAMLIT INTERFACE ==================

def main():

    # Load CSS
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    #st.set_page_config(
       # page_title="Waste Collection Optimization",
        #page_icon="🗑️",
        #layout="wide"
    #)
    
    # Initialize session state
    if 'optimization_data' not in st.session_state:
        st.session_state.optimization_data = {
            'complete': False,
            'solution': None,
            'fitness_history': [],
            'distance_history': [],
            'map_html': None,
            'stats': None,
            'point_list': None,
            'parc': None,
            'cet': None,
            'camions_dict': None
        }
    
    st.title("🗑️ Waste Collection Route Optimization")
    st.markdown("Optimize waste collection routes using genetic algorithms")
    
   

   


    # Sidebar controls
    with st.sidebar:
        st.header("Parameters")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        generations = st.slider("Generations", 10, 100, 30)
        population_size = st.slider("Population", 5, 50, 15)
        run_button = st.button("Run Optimization", type="primary")
        reset_button = st.button("Reset Results")
    

    if reset_button:
        st.session_state.optimization_data = {
            'complete': False,
            'solution': None,
            'fitness_history': [],
            'distance_history': [],
            'map_html': None,
            'stats': None,
            'point_list': None,
            'parc': None,
            'cet': None,
            'camions_dict': None
        }
        st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Progress", "Map", "Details"])
    
    if run_button:
        with st.spinner("Optimizing routes..."):
            try:
                df = load_and_preprocess_data(uploaded_file if uploaded_file else "bdd_zone_A.csv")
                camions, parc, cet, points_collecte, camions_dict, points_dict = extract_entities(df)
                point_list = [points_dict[num] for num in sorted(points_dict.keys())]
                precompute_distances(point_list, parc, cet)
                
                # Convert camions_dict.values() to list for indexing
                trucks_list = list(camions_dict.values())
                
                start_time = time.time()
                best_solution, fitness_history, distance_history = run_streamlit_genetic_algorithm(
                    generations, population_size, 0.8, 0.1, point_list, parc, cet, camions_dict
                )
                elapsed_time = time.time() - start_time
                
                # Calculate stats
                total_distance = sum(calculate_route_distance_fast(r['points'], point_list, parc, cet) for r in best_solution)
                total_load = sum(sum(point_list[p-1].stock_initial_t for p in r['points']) for r in best_solution)
                truck_usage = defaultdict(int)
                for route in best_solution:
                    truck_usage[route["truck"]] += 1
                
                # Create map
                m = folium.Map(location=[parc.latitude, parc.longitude], zoom_start=13)
                colors = ['red', 'blue', 'green', 'purple', 'orange']
                
                # Add depot and landfill
                folium.Marker(
                    [parc.latitude, parc.longitude],
                    popup='Depot',
                    icon=folium.Icon(color='black', icon='home')
                ).add_to(m)
                
                folium.Marker(
                    [cet.latitude, cet.longitude],
                    popup='Landfill',
                    icon=folium.Icon(color='red', icon='trash')
                ).add_to(m)
                
                # Add collection points
                for i, point in enumerate(point_list, 1):
                    folium.CircleMarker(
                        [point.latitude, point.longitude],
                        radius=5,
                        popup=f"Point {point.numero}",
                        color='gray',
                        fill=True
                    ).add_to(m)
                
                # Add routes
                for i, route in enumerate(best_solution):
                    route_points = [parc] + [point_list[p-1] for p in route["points"]] + [cet]
                    folium.PolyLine(
                        [(p.latitude, p.longitude) for p in route_points],
                        color=colors[i % len(colors)],
                        weight=3,
                        popup=f"Route {i+1}"
                    ).add_to(m)
                
                # Save results
                st.session_state.optimization_data = {
                    'complete': True,
                    'solution': best_solution,
                    'fitness_history': fitness_history,
                    'distance_history': distance_history,
                    'map_html': m._repr_html_(),
                    'stats': {
                        'total_distance': total_distance,
                        'total_load': total_load,
                        'truck_usage': dict(truck_usage),
                        'time': elapsed_time
                    },
                    'point_list': point_list,
                    'parc': parc,
                    'cet': cet,
                    'camions_dict': camions_dict
                }
                
                st.success(f"Optimization complete in {elapsed_time:.1f} seconds!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display results
    if st.session_state.optimization_data['complete']:
        data = st.session_state.optimization_data
        
        with tab1:
            st.header("Optimization Progress")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(data['fitness_history'])
            ax1.set_title("Fitness Convergence")
            ax2.plot(data['distance_history'])
            ax2.set_title("Distance Optimization")
            st.pyplot(fig)
            st.metric("Total Time", f"{data['stats']['time']:.1f} seconds")
        
        with tab2:
            st.header("Optimized Routes")
            st.components.v1.html(data['map_html'], height=600)
            with open("temp_map.html", "w") as f:
                f.write(data['map_html'])
            with open("temp_map.html", "rb") as f:
                st.download_button(
                    "Download Map",
                    f,
                    "optimized_routes.html",
                    "text/html"
                )
        
        with tab3:
            st.header("Solution Details")
            st.write(f"**Total Distance:** {data['stats']['total_distance']:.1f} km")
            st.write(f"**Total Waste Collected:** {data['stats']['total_load']:.1f} t")
            
            st.subheader("Truck Usage")
            for truck, count in data['stats']['truck_usage'].items():
                st.write(f"- Truck {truck}: {count} routes")
            
            st.subheader("Routes")
            for i, route in enumerate(data['solution']):
                with st.expander(f"Route {i+1} - Truck {route['truck']}"):
                    st.write(f"Distance: {calculate_route_distance_fast(route['points'], data['point_list'], data['parc'], data['cet']):.1f} km")
                    st.write(f"Load: {sum(data['point_list'][p-1].stock_initial_t for p in route['points']):.1f} t")
                    st.write(f"Points: {', '.join(str(p) for p in route['points'])}")

                    
if __name__ == "__main__":
    main()