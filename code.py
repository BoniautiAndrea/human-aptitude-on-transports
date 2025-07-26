import numpy as np
import shortestpaths as sp
import networkx as nx

# A graph, representing a street map highlighting routes available for cars, buses or metro
# nodes not so important, they represent exchange points for different type of transport
# in the typed graph edges represent routes, they can be for cars, bus or metro
#
# in time graph, edges represent time needed for that route
#
# user defined using Big-Five traits
# user example: u = [Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness] wih value {0, 1}
# pure car drivers traits: [1, 0, 1, 0, 0] (typical initial state)
# pure PT commuters traits: [0, 1, 0, 1, 1] (typical objective state)
#
# affinity of arcs is calculated combining user features, arc type and time needed
# after a planning iteration, a feedback is expected.
# a feedback modifies user features, and then the graph is computed again with the new affinity.
#
# Given that the theoretical goal is to convert a car driver into a public commuter one
# for a positive feedback, high degree traits of a pure car driver will be downgraded
# while high degree traits of a pure PT commuter will be upgraded
# for a negative feedback it'll be the opposite

np.set_printoptions(suppress=True)

# Building the graph

#adjacency matrix that defines type of arcs
# 1 = metro, 2 = bus, 3 = car
type_graph = np.array([
    [0, 2, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
    [2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
    [0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
    [3, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
    [0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #6
    [0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0], #7
    [0, 0, 0, 3, 0, 0, 0, 2, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0], #8
    [0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0], #9
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0], #10
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0], #11
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 3, 3, 0, 0], #12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 2, 0, 3, 0, 3, 2, 1], #13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 3], #14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0], #15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 2, 0], #16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2], #17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 0]  #18
])

#adjacency matrix that defines costs/time for each arc
time_graph = np.array([
    [0, 4, 0, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
    [5, 0, 7, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
    [0, 2, 0, 0, 0, 7, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
    [3, 0, 0, 0, 0, 0, 0, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
    [0, 7, 4, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
    [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], #6
    [0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0], #7
    [0, 0, 0, 8, 0, 0, 0, 6, 0, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0], #8
    [0, 0, 0, 0, 6, 2, 0, 0, 5, 0, 4, 0, 0, 6, 0, 0, 0, 0, 0], #9
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 3, 4, 0, 0, 0, 0], #10
    [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0], #11
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 7, 0, 4, 6, 0, 0], #12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 0, 4, 0, 7, 0, 2, 4, 4], #13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 6, 0, 0, 0, 0, 4], #14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0], #15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 4, 0], #16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 6, 0, 5], #17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 5, 0, 0, 4, 0]  #18
])

# Computes affinity (TODO: differentiate affinity for bus and metro)
def affinity(type, time, user):
    if type == 3:
        return time*(np.power(2, (-1+2*abs(sum(np.subtract(user, [1, 0, 1, 0, 0])))), dtype=float))
    else:
        return time*(np.power(2, (-1+2*abs(sum(np.subtract(user, [0, 1, 0, 1, 1])))), dtype=float))

# Builds a new graph with affinity
def compute_graph(user, type_graph, time_graph):
    weight_graph = np.zeros((19,19))
    for i in range(19):
        for j in range(19):
            weight_graph[i][j] = affinity(type_graph[i][j], time_graph[i][j], user)
    return weight_graph

# Update user preferences
def apply_feedback(user, feedback):
    if feedback == 1:
        user = np.add(user, [-0.1, 0.1, -0.1, 0.1, 0.1])
    else:
        user = np.add(user, [0.1, -0.1, 0.1, -0.1, -0.1])
    print('Feedback applied')
    print('New user preferences according to Big five: ' + str(user))
    return user

# Main method, creates initial graph and then plan-feedback-update loop
def planning_loop(user, type_graph, time_graph, feedback, iters):
    graph = compute_graph(user, type_graph, time_graph)
    print('Graph generated')

    for i in range(iters):
        print('Starting planning num. ' + str(i+1))
        g = nx.Graph(graph)
        # Take the second shortest path to force the user to move away from its preferences
        path, cost = sp.k_shortest_paths(g, 0, 18, 2)[1]
        print('Path generated: ' + str(path))

        user = apply_feedback(user, feedback[i])
        graph = compute_graph(user, type_graph, time_graph)
        print('Graph updated')
    
    print('Loop finished')
    print('Final user values: ' + str(user))


# Starting user traits, we use typical car driver traits as start to transform it into a pure commuter: [0, 1, 0, 1, 1]
user = [1, 0, 1, 0, 0]

# All positives just to see the transformation ASAP
feedback = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

#
planning_loop(user, type_graph, time_graph, feedback, 10)
