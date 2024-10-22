import time
import heapq
import matplotlib.pyplot as plt

# A* algorithm for N=5, K=3
class State:
    def __init__(self, missionaries_left, cannibals_left, boat_left, missionaries_right, cannibals_right, boat_right):
        self.m_left = missionaries_left
        self.c_left = cannibals_left
        self.boat_left = boat_left
        self.m_right = missionaries_right
        self.c_right = cannibals_right
        self.boat_right = boat_right

    def is_valid(self):
        # Missionaries cannot be outnumbered by cannibals on either side
        if (self.m_left < self.c_left and self.m_left > 0) or (self.m_right < self.c_right and self.m_right > 0):
            return False
        return True

    def is_goal(self):
        # Goal state: all missionaries and cannibals are on the right bank
        return self.m_left == 0 and self.c_left == 0

    def __lt__(self, other):
        return True

    def __repr__(self):
        return f"({self.m_left}, {self.c_left}, {self.boat_left}, {self.m_right}, {self.c_right}, {self.boat_right})"

# Heuristic function 1: based on remaining people
def heuristic_1(state,_):
    balance_penalty = 0
    if state.m_left < 0 or state.c_left < 0 or state.c_right < 0 or state.m_right < 0:
        balance_penalty += 10
    if state.m_left > 0 and state.m_left < state.c_left:
        balance_penalty += 10  # Apply a higher penalty if the state is highly imbalanced
    return (state.m_left + state.c_left) // 3 + balance_penalty  # Adjust for boat capacity K=3

# Heuristic function 2: based on remaining people, balance, and depth
def heuristic_2(state, depth):
    balance_penalty = 0
    if state.m_left < 0 or state.c_left < 0 or state.c_right < 0 or state.m_right < 0:
        balance_penalty += 10
    if state.m_left > 0 and state.m_left < state.c_left:
        balance_penalty += 10  # Apply a higher penalty if the state is highly imbalanced
    # Add depth as a penalty
    depth_penalty = depth * 0.2 # You can adjust the weight of depth here
    return (state.m_left + state.c_left) // 3 + balance_penalty + depth_penalty


# Heuristic function 3: based on remaining people, balance, depth, and the new penalty function (M + C - K * B)
def heuristic_3(state, depth):
    balance_penalty = 0
    if state.m_left < 0 or state.c_left < 0 or state.c_right < 0 or state.m_right < 0:
        balance_penalty += 10
    if state.m_left > 0 and state.m_left < state.c_left:
        balance_penalty += 10  # Apply a higher penalty if the state is highly imbalanced

    # Add depth as a penalty
    depth_penalty = depth * 0.2  # You can adjust the weight of depth here

    # Add the new penalty function: h(n) = M + C - K * B
    penalty_function = state.m_left + state.c_left - 3 * state.boat_left

    # Final heuristic value
    return  balance_penalty + depth_penalty + penalty_function


# A* search implementation with state printing and depth tracking
def a_star(start_state, heuristic):
    open_list = []
    heapq.heappush(open_list, (heuristic(start_state, 0), start_state, 0))  # Include depth in the heap (initially 0)
    visited = set()

    while open_list:
        _, current_state, depth = heapq.heappop(open_list)  # Extract the state and its depth

        # Print the current state
        print(f"Current state at depth {depth}: {current_state}")

        if current_state.is_goal():
            print("Goal state reached!")
            return True  # Goal found

        if repr(current_state) in visited:
            continue

        visited.add(repr(current_state))

        # Generate next possible states for K=3
        moves = [(3, 0), (2, 1), (1, 2), (0, 3), (2, 0), (1, 1), (0, 2), (1, 0), (0, 1)]
        for m, c in moves:
            if current_state.boat_left:
                new_state = State(current_state.m_left - m, current_state.c_left - c, 0,
                                  current_state.m_right + m, current_state.c_right + c, 1)
            else:
                new_state = State(current_state.m_left + m, current_state.c_left + c, 1,
                                  current_state.m_right - m, current_state.c_right - c, 0)

            if new_state.is_valid() and repr(new_state) not in visited:
                print(f"Generated new state: {new_state} at depth {depth + 1}")
                heapq.heappush(open_list, (heuristic(new_state, depth + 1), new_state, depth + 1))

    print("No solution found.")
    return False  # No solution found


# Run and compare heuristic functions for N=5, K=3
def compare_heuristics():
    times_heuristic_1 = []
    times_heuristic_2 = []
    times_heuristic_3 = []
    epoch = 10
    for i in range(epoch):
        start_state = State(5, 5, 1, 0, 0, 0)  # Start with 5 missionaries and 5 cannibals


        print(f"Running iteration {i+1} for heuristic 1...")
        start_time = time.time()
        a_star(start_state, heuristic_1)
        times_heuristic_1.append(time.time() - start_time)

        print(f"Running iteration {i+1} for heuristic 2...")
        start_time = time.time()
        a_star(start_state, heuristic_2)
        times_heuristic_2.append(time.time() - start_time)

        print(f"Running iteration {i+1} for heuristic 3...")
        start_time = time.time()
        a_star(start_state, heuristic_3)
        times_heuristic_3.append(time.time() - start_time)

    # Visualize results
    plt.plot(range(1, epoch+1), times_heuristic_1, label="Heuristic 1", marker='o')
    plt.plot(range(1, epoch+1), times_heuristic_2, label="Heuristic 2", marker='x')
    plt.plot(range(1, epoch + 1), times_heuristic_3, label="Heuristic 3", marker='s')
    plt.xlabel("Run Number")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of Heuristic 1 and Heuristic 2 Execution Time for N=5, K=3")
    plt.legend()
    plt.show()

# Execute the comparison
compare_heuristics()
