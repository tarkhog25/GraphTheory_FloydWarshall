"""
Floyd-Warshall Algorithm
SM601I - Graph Theory Project
Year 2025/2026 - Department of Mathematics
"""

import math
import os

#  Constants
INF = math.inf

#  1. FILE LOADING

def load_graph(filename):
    """
    Load a graph from a text file with the following format:
      Line 1  : number of vertices
      Line 2  : number of edges
      Lines 3+: <source> <destination> <weight>

    Returns (n, edges) where n is the number of vertices and
    edges is a list of (source, destination, weight) tuples.
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    nb_edges = int(lines[1])

    edges = []
    for i in range(2, 2 + nb_edges):
        parts = lines[i].split()
        src, dst, w = int(parts[0]), int(parts[1]), int(parts[2])
        edges.append((src, dst, w))

    return n, edges


#  2. MATRIX REPRESENTATION

def build_matrix(n, edges):
    """
    Build the initial weight matrix from the edge list.
    - L[i][i] = 0 (diagonal of 0)
    - L[i][j] = weight  (if edge (i,j) exists)
    - L[i][j] = INF     (if not)
    """
    L = [[INF] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 0
    for (src, dst, w) in edges:
        L[src][dst] = w
    return L


#  3. DISPLAY HELPERS

def format_val(v, width):
    """Format a matrix value for display (INF → '+inf')."""
    if v == INF:
        s = "+inf"
    else:
        s = str(int(v)) if v == int(v) else str(v)
    return s.rjust(width)


def print_matrix_L(L, n, title):
    """Print an L matrix with row/column headers."""
    # Determine column width
    vals = []
    for i in range(n):
        for j in range(n):
            vals.append(len(format_val(L[i][j], 1)))
    col_w = max(max(vals), len(str(n - 1))) + 1

    header_w = len(str(n - 1)) + 1

    print(f"\n  {title}")
    # Column header
    header = " " * (header_w + 2)
    for j in range(n):
        header += str(j).rjust(col_w)
    print(header)
    # Separator
    print(" " * (header_w + 2) + "-" * (col_w * n))
    # Rows
    for i in range(n):
        row = str(i).rjust(header_w) + " |"
        for j in range(n):
            row += format_val(L[i][j], col_w)
        print(row)


def print_matrix_P(P, n, title):
    """Print a P (predecessor) matrix with row/column headers."""
    col_w = max(len(str(n - 1)), 4) + 1
    header_w = len(str(n - 1)) + 1

    print(f"\n  {title}")
    header = " " * (header_w + 2)
    for j in range(n):
        header += str(j).rjust(col_w)
    print(header)
    print(" " * (header_w + 2) + "-" * (col_w * n))
    for i in range(n):
        row = str(i).rjust(header_w) + " |"
        for j in range(n):
            v = P[i][j]
            s = str(v) if v is not None else "—"
            row += s.rjust(col_w)
        print(row)


def display_weight_matrix(L, n):
    """Display the initial weight matrix of the graph."""
    print("\n" + "=" * 60)
    print("  WEIGHT MATRIX (initial)")
    print("=" * 60)
    print_matrix_L(L, n, "L⁰")


#  4. FLOYD-WARSHALL ALGORITHM

def floyd_warshall(L0, n):
    """
    L[i][j] stores the shortest known distance from i to j.
    P[i][j] stores the predecessor of j on the current shortest path from i to j.

    At initialisation:
      - P[i][j] = i  if there is a direct edge i→j or i==j
      - P[i][j] = None  if L[i][j] == INF (no path known)

    Update: when L[i][k] + L[k][j] < L[i][j],
      set P[i][j] ← P[k][j]   (predecessor of j on path via k)

    Returns (L_final, P_final, absorbing_circuit_detected)
    Also prints intermediate matrices at each step k.
    """
    # Deep copy
    L = [row[:] for row in L0]

    # Initialise P
    P = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if L[i][j] != INF:
                P[i][j] = i   # direct edge or self-loop

    print("\n" + "=" * 60)
    print("  FLOYD-WARSHALL  —  INTERMEDIATE STEPS")
    print("=" * 60)
    print_matrix_L(L, n, "L⁰  (initial)")
    print_matrix_P(P, n, "P⁰  (initial)")

    absorbing = False

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if L[i][k] != INF and L[k][j] != INF:
                    new_dist = L[i][k] + L[k][j]
                    if new_dist < L[i][j]:
                        L[i][j] = new_dist
                        P[i][j] = P[k][j]

        # Check for absorbing circuit
        for i in range(n):
            if L[i][i] < 0:
                absorbing = True

        print(f"\n  ── After iteration k = {k} ──")
        print_matrix_L(L, n, f"L^{k+1}")
        print_matrix_P(P, n, f"P^{k+1}")

        if absorbing:
            print("\n  ⚠  ABSORBING CIRCUIT found.")
            print("     Stopping early.")
            break

    return L, P, absorbing



#  5. PATH RECONSTRUCTION

def reconstruct_path(P, src, dst, n):
    """
    Reconstruct the shortest path from src to dst using matrix P.
    Returns a list of vertices [src, intermediate vertices, dst], or None if no path.
    """
    if P[src][dst] is None:
        return None   # no path

    path = [dst]
    current = dst
    visited = set()

    while current != src:
        if current in visited:
            return None  # cycle guard (should not happen without absorbing circuit)
        visited.add(current)
        pred = P[src][current]
        if pred is None:
            return None
        path.append(pred)
        current = pred

    path.reverse()
    return path


def display_all_paths(L, P, n):
    """Display all shortest paths between every pair of vertices."""
    print("\n" + "=" * 60)
    print("  ALL SHORTEST PATHS")
    print("=" * 60)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            path = reconstruct_path(P, i, j, n)
            dist = L[i][j]
            if path is None:
                print(f"  {i} → {j} :  No path")
            else:
                path_str = " → ".join(str(v) for v in path)
                print(f"  {i} → {j} :  {path_str}   (length = {int(dist)})")


def interactive_path_query(L, P, n):
    """Let the user query individual shortest paths interactively."""
    print("\n" + "=" * 60)
    print("  PATH QUERY MODE")
    print("=" * 60)
    while True:
        print()


        src = int(input(f"  Starting vertex (0-{n-1}): ").strip())
        dst = int(input(f"  Ending   vertex (0-{n-1}): ").strip())

        if not (0 <= src < n and 0 <= dst < n):
            print(f"  Vertices must be between 0 and {n-1}.")
            continue

        if src == dst:
            print(f"  {src} → {dst} : same vertex (distance = 0)")
            continue

        path = reconstruct_path(P, src, dst, n)
        dist = L[src][dst]
        if path is None:
            print(f"  {src} → {dst} : No path exists.")
        else:
            path_str = " → ".join(str(v) for v in path)
            print(f"  {src} → {dst} :  {path_str}   (length = {int(dist)})")

        again = input("  Another path? (yes/no) : ").strip().lower()
        if again not in ("yes", "y", "oui", "o"):
            break


#  6. PROCESS ONE GRAPH

def process_graph(graph_number):
    """Load, analyse and display results for one graph."""
    filename = f"graph{graph_number}.txt"

    # ── (1) User specified the number; build filename
    print(f"\n{'#' * 60}")
    print(f"  GRAPH {graph_number}  ←  {filename}")
    print(f"{'#' * 60}")

    # ── (2) Load from file
    if not os.path.exists(filename):
        print(f"  ERROR: file '{filename}' not found.")
        return
    try:
        n, edges = load_graph(filename)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return

    print(f"\n  {n} vertices,  {len(edges)} edges loaded.")

    # in-memory structure
    L0 = build_matrix(n, edges)

    # Display initial matrix
    display_weight_matrix(L0, n)

    # Run Floyd-Warshall
    L_final, P_final, absorbing = floyd_warshall(L0, n)

    # Report absorbing circuit
    print("\n" + "=" * 60)
    if absorbing:
        print("  RESULT: The graph CONTAINS at least one absorbing circuit.")
        print("          Shortest paths cannot be computed reliably.")
    else:
        print("  RESULT: The graph contains NO absorbing circuit.")

        # Display paths
        print("\n  How would you like to display the shortest paths?")
        print("    1 - Display ALL pairs")
        print("    2 - Query paths interactively")
        choice = input("  Your choice (1/2): ").strip()
        if choice == "1":
            display_all_paths(L_final, P_final, n)
        else:
            interactive_path_query(L_final, P_final, n)

    print("=" * 60)



#  7. MAIN LOOP


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Floyd-Warshall  —  Shortest Path                 ║")
    print("║        SM601I Graph Theory  ·  Mohammed & Maxime         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    while True:
        print()
        raw = input("Enter graph number to analyse (or 'quit' to exit): ").strip()
        if raw.lower() in ("quit", "q", "exit"):
            print("Goodbye.")
            break
        try:
            graph_number = int(raw)
        except ValueError:
            print("  Please enter a valid integer.")
            continue
        process_graph(graph_number)


if __name__ == "__main__":
    main()
