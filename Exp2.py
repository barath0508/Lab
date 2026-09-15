# A* Search Algorithm

Graph = {
    'A': [('B', 6), ('F', 3)],
    'B': [('A', 6), ('C', 3), ('D', 2)],
    'C': [('B', 3), ('D', 1), ('E', 5)],
    'D': [('B', 2), ('C', 1), ('E', 8)],
    'E': [('C', 5), ('D', 8), ('I', 5), ('J', 5)],
    'F': [('A', 3), ('G', 1), ('H', 7)],
    'G': [('F', 1), ('I', 3)],
    'H': [('F', 7), ('I', 2)],
    'I': [('E', 5), ('G', 3), ('H', 2), ('J', 3)],
    'J': []
}

H = {
    'A': 11, 'B': 6, 'C': 5, 'D': 7, 'E': 3,
    'F': 6, 'G': 5, 'H': 3, 'I': 1, 'J': 0
}

def a_star(start, goal):
    open_list = [start]
    cost = {start: 0}
    parent = {start: None}

    while open_list:
        # Select node with minimum f = g + h
        current = min(open_list,
                      key=lambda x: cost[x] + H[x])

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]

            path.reverse()
            print("Path found:", path)
            print("Cost:", cost[goal])
            return

        open_list.remove(current)

        for node, weight in Graph[current]:
            new_cost = cost[current] + weight

            if node not in cost or new_cost < cost[node]:
                cost[node] = new_cost
                parent[node] = current

                if node not in open_list:
                    open_list.append(node)

    print("Path does not exist!")


a_star('A', 'J')