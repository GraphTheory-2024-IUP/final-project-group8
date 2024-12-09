[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/0_yE0bFY)
# Segmentation Simplified: Using Graph Cut Techniques for Image Divisions

| Name           | NRP        | Kelas     |
| ---            | ---        | ----------|
| Kinar Naila Fauziah | 5025231001| Theory Graph (IUP)  |
| Sinta Prabondari Wardani| 5025231067 | Theory Graph (IUP) |
| Azkiya Rusyda Zahra |  5025231072 | Theory Graph (IUP) |
| Safa Nadhira  Azzahra | 5025231086  | Theory Graph (IUP) |


### Introduction 
Image segmentation divides an image into meaningful regions, crucial for tasks like autonomous navigation, medical imaging, photo editing, and object detection. Graph cut techniques are highly effective for segmentation, treating each pixel as a graph node connected by edges based on attributes like color or texture. This transforms segmentation into a graph partitioning problem, aiming to minimize the "cut" cost, which yields precise, adaptable boundaries suited for complex shapes and lighting. In "Segmentation Simplified: Using Graph Cut Techniques for Image Divisions," we examine how graph cuts offer an intuitive, flexible approach to overcoming segmentation challenges across various applications.


### Program Demonstration

## Segmentation output

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











