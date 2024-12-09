import numpy as np
import networkx as nx


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


def label_regions(grid, regions):
    """
    Assign labels to the regions and create a labeled grid.
    """
    labeled_grid = np.full(grid.shape, '', dtype=str)
    for label, region in enumerate(regions):
        for (r, c) in region:
            labeled_grid[r, c] = chr(65 + label)  # Assign letters A, B, C...

    return labeled_grid


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


# Main function
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
