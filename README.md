[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/0_yE0bFY)
# Segmentation Simplified: Using Graph Cut Techniques for Image Divisions

| Name           | NRP        | Kelas     |
| ---            | ---        | ----------|
| Kinar Naila Fauziah | 5025231001| Theory Graph (IUP)  |
| Sinta Prabondari Wardani| 5025231067 | Theory Graph (IUP) |
| Azkiya Rusyda Zahra |  5025231072 | Theory Graph (IUP) |
| Safa Nadhira  Azzahra | 5025231086  | Theory Graph (IUP) |


## Introduction 
Image segmentation divides an image into meaningful regions, crucial for tasks like autonomous navigation, medical imaging, photo editing, and object detection. Graph cut techniques are highly effective for segmentation, treating each pixel as a graph node connected by edges based on attributes like color or texture. This transforms segmentation into a graph partitioning problem, aiming to minimize the "cut" cost, which yields precise, adaptable boundaries suited for complex shapes and lighting. In "Segmentation Simplified: Using Graph Cut Techniques for Image Divisions," we examine how graph cuts offer an intuitive, flexible approach to overcoming segmentation challenges across various applications.


## Program Demonstration

### Code 1
### Segmentation output

```py
def grid_to_graph(grid):
    """
    Convert a 2D grid into a graph where each cell is a node,
    and edges are weighted by differences between adjacent cells.
    """
    rows, cols = grid.shape
    graph = nx.Graph()

    for r in range(rows):
        for c in range(cols):
            node = (r, c)
            graph.add_node(node, value=grid[r, c])

            # Connect to the right and bottom neighbors
            if c + 1 < cols:  # Right neighbor
                weight = abs(grid[r, c] - grid[r, c + 1])
                graph.add_edge(node, (r, c + 1), weight=weight)
            if r + 1 < rows:  # Bottom neighbor
                weight = abs(grid[r, c] - grid[r + 1, c])
                graph.add_edge(node, (r + 1, c), weight=weight)

    return graph
```
This converts a 2D grid into a graph, where each cell is a node, and edges are weighted based on the value difference between adjacent cells.
- Flow
  1. The dimension of the grid (`rows` and `cols`) will be extracted
  2. A graph will be initialized using NetworkX
  3. FOr each cell `(r, c)` in the grid:
     - A node will be added to the graph, tagged with its value from the grid
     - Edges are created (with the weight of each edge is the absolute difference between the values of the connected cells)
  4. The graph is returned


```py
def segment_grid_with_mst(graph, threshold):
    """
    Perform grid segmentation using the MST method with a threshold.
    """
    mst = nx.minimum_spanning_tree(graph)
    segmented_graph = nx.Graph()
    segmented_graph.add_nodes_from(mst.nodes(data=True))

    # Add only edges with weight <= threshold
    for u, v, data in mst.edges(data=True):
        if data['weight'] <= threshold:
            segmented_graph.add_edge(u, v, weight=data['weight'])

    # Find connected components as distinct regions
    regions = list(nx.connected_components(segmented_graph))
    return regions
```
This function segments the grid into regions using MST
- FLow
  1. Compute the MST of the graph
  2. Create a new graph (`segmented_graph`) to store segments
  3. Add nodes from the MST to the new graph
  4. Edges will be added to the segmented graph only if their weight is <= threshold
  5. Extract the connected components (regions) from the segmented graph (each connected component represents a distinct region)
  6. Finally, it will return the list of regions


```py
def label_regions(grid, regions):
    """
    Assign labels to the regions and create a labeled grid.
    """
    labeled_grid = np.full(grid.shape, '', dtype=str)
    for label, region in enumerate(regions):
        for (r, c) in region:
            labeled_grid[r, c] = chr(65 + label)  # Assign letters A, B, C...

    return labeled_grid
```
this function will label the segmented regions on a grid
- Flow
  1. First, it will initialize a new grid (`labeled_grid`)
  2. For each region, it will assign a unique label (letters starting from A)
  3. Then it will iterate through each node in a region and mark its position in the `labeled_grid` with the region's label
  4. Finally return the labeled grid
 

```py
def read_input():
    """
    Reads a custom 2D grid from user input.
    """
    print("Enter the number of rows:")
    rows = int(input())
    print("Enter the number of columns:")
    cols = int(input())

    print(f"Enter the grid values row by row (space-separated):")
    grid = []
    for i in range(rows):
        print(f"Row {i + 1}:")
        row = list(map(int, input().split()))
        grid.append(row)

    return np.array(grid)
```
This will read a 2D grid from user's input
- FLow
  1. It will prompt the user to enter the number of rows & columns
  2. Then it'll collect the grid values row by row, then parsing them into integers
  3. and finally returns the grid as numpy array

 
```py
if __name__ == "__main__":
    print("Welcome to the Grid Segmentation Program!")

    # Input custom grid
    input_grid = read_input()

    # Set threshold for segmentation
    print("Enter the threshold value for segmentation:")
    threshold = int(input())

    # Process
    graph = grid_to_graph(input_grid)
    regions = segment_grid_with_mst(graph, threshold)
    labeled_grid = label_regions(input_grid, regions)

    # Output
    print("\nInput Grid:")
    print(input_grid)

    print("\nSegmented Grid (Labeled):")
    for row in labeled_grid:
        print(" ".join(row))
```
The main function will be the function where we call other functions

This are an example Input and Output
![image](https://github.com/user-attachments/assets/431e1894-892d-4be8-a00d-b2d3ed0fa389)



### Image output

```py
def rgb_to_grayscale(image):
    """Convert a color image (RGB) to grayscale by averaging color channels."""
    return np.mean(image, axis=2).astype(np.int32)
```
this function converts an RGB image into a grayscale image by averaging the red, green, and blue channels


```py
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
```
This converts the grayscale image into a graph. Each pixel becomes a node, while the edge represents the intensity difference between adjacent pixels 


```py
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
```
This function categorizes edge weights into predefined intensity difference categories. it classifies weights into group based on their range


```py
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
```
The function segments the image using MST methods and assigns categories to edges
- Flow
  1. First, it converts the grayscale image into a graph
  2. it sorts all edges by weight
  3. Next, it will build an MST
     - Where it uses kruskal's algorithm with union-find data structures for cycle detection
     - Then it will add edges to the MST only if their weight is below the threshold
  4. After, it will find connected components in the MST to define regions
  5. Last, it assigns a category to each edge based on its weight classifications


```py
def visualize_segmented_colors(labels):
    """Visualize the segmented regions with color categories."""
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = np.vectorize(label_to_int.get)(labels)
    
    plt.imshow(int_labels, cmap='tab10')  # Map categories to distinct colors
    plt.axis('off')
    plt.show()
```
This function is to visualize the segmented image with distinct colors for each category
- Flow
  1. Converts label categories into integers for visualization
  2. Then, displays the image using a colormap to represent categories as colors
 

```py
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
```
Finally, the main function is to run the entire process, from image loading to visualization
- Flow
  1. Loads and preprocess the image
     - It loads an image file and converts it to RGB
     - then it will resize the image to 75x75 for simplicity
     - Converts the resized image to grayscale
  2. Set segmentation threshold (it will define the intensity difference threshold for segmentation)
  3. Segment the image
  4. Visualize the result




