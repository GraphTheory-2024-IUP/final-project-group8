from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def rgb_to_grayscale(image):
    """Convert a color image (RGB) to grayscale by averaging color channels."""
    return np.mean(image, axis=2).astype(np.int32)

def image_to_graph(image):
    """Convert a 2D grayscale image to a graph with pixel intensity differences as edge weights."""
    rows, cols = image.shape
    graph = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            graph.add_node((r, c), intensity=int(image[r, c]))
            if r > 0:  # Vertical edge
                weight = abs(int(image[r, c]) - int(image[r - 1, c]))
                graph.add_edge((r, c), (r - 1, c), weight=weight)
            if c > 0:  # Horizontal edge
                weight = abs(int(image[r, c]) - int(image[r, c - 1]))
                graph.add_edge((r, c), (r, c - 1), weight=weight)
    return graph

def classify_edge_weight(weight):
    """Classify edge weight into categories based on intensity."""
    if 1 <= weight <= 10:
        return "A"
    elif 11 <= weight <= 20:
        return "B"
    elif 21 <= weight <= 30:
        return "C"
    elif 31 <= weight <= 40:
        return "D"
    elif 41 <= weight <= 50:
        return "E"
    elif 51 <= weight <= 60:
        return "F"
    elif 61 <= weight <= 70:
        return "G"
    else:
        return "H"  # Default for weights > 70

def segment_image_with_color_categories(image, threshold):
    """Segment the image and categorize edges into color groups."""
    # Step 1: Convert image to graph
    graph = image_to_graph(image)
    
    # Step 2: Sort edges by weight
    edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
    
    # Step 3: Build MST
    mst = nx.Graph()
    parent = {}
    rank = {}

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            elif rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    for node in graph.nodes:
        parent[node] = node
        rank[node] = 0

    edge_colors = {}  # Store edge classifications (A, B, C, etc.)
    for u, v, attr in edges:
        weight = attr['weight']
        edge_color = classify_edge_weight(weight)
        edge_colors[(u, v)] = edge_color  # Store edge's classification
        
        if find(u) != find(v):
            if weight <= threshold:
                mst.add_edge(u, v, weight=weight)
                union(u, v)
    
    # Step 4: Extract connected components as segments
    labels = np.zeros(image.shape, dtype=str)
    for idx, component in enumerate(nx.connected_components(mst)):
        for node in component:
            labels[node] = list(edge_colors.values())[idx]  # Assign corresponding color
    
    return labels

def visualize_segmented_colors(labels):
    """Visualize the segmented regions with color categories."""
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = np.vectorize(label_to_int.get)(labels)
    
    plt.imshow(int_labels, cmap='tab10')  # Map categories to distinct colors
    plt.axis('off')
    plt.show()

def main():
    # Load and preprocess the image
    image_path = 'Downloads/shocked.jpg'  # Replace with your image path
    img = Image.open(image_path).convert('RGB')
    img = img.resize((75, 75))  # Resize for simplicity
    grayscale_image = rgb_to_grayscale(np.array(img))
    
    # Define threshold
    threshold =  5 # Adjust as needed
    
    # Segment image with color categories
    segmented_labels = segment_image_with_color_categories(grayscale_image, threshold)
    
    # Visualize segmented regions with color categories
    visualize_segmented_colors(segmented_labels)

if __name__ == "__main__":
    main()
