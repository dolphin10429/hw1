from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

DISCOUNT_FACTOR = 0.95  # 折扣因子
MAX_ITERATIONS = 1000  # 最大迭代次數
CONVERGENCE_THRESHOLD = 1e-9  # 收斂閾值

def evaluate_policy(grid_size, policy_matrix, obstacles, start_point, end_point):
    value_matrix = np.zeros((grid_size, grid_size), dtype=object)

    for (i, j) in obstacles:
        value_matrix[i][j] = None

    def is_within_bounds(i, j):
        return 0 <= i < grid_size and 0 <= j < grid_size

    def get_next_state(i, j):
        if (i, j) in obstacles:
            return (i, j), 0

        action = policy_matrix[i][j]
        if action == '■':
            return (i, j), 0

        if action == '↑':
            new_i, new_j = i - 1, j
        elif action == '↓':
            new_i, new_j = i + 1, j
        elif action == '←':
            new_i, new_j = i, j - 1
        elif action == '→':
            new_i, new_j = i, j + 1
        else:
            return (i, j), 0

        if not is_within_bounds(new_i, new_j) or (new_i, new_j) in obstacles:
            return (i, j), -1
        else:
            return (new_i, new_j), 0

    for _ in range(MAX_ITERATIONS):
        delta = 0
        updated_values = value_matrix.copy()

        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in obstacles:
                    continue

                (next_i, next_j), reward = get_next_state(i, j)

                if (next_i, next_j) in obstacles:
                    next_value = 0
                else:
                    next_value = value_matrix[next_i][next_j]
                    if next_value is None:
                        next_value = 0

                updated_value = reward + DISCOUNT_FACTOR * next_value
                current_value = value_matrix[i][j]
                if current_value is None:
                    current_value = 0
                value_diff = abs(updated_value - current_value)
                if value_diff > delta:
                    delta = value_diff

                updated_values[i][j] = round(updated_value, 2)

        value_matrix = updated_values
        if delta < CONVERGENCE_THRESHOLD:
            break

    return value_matrix

def generate_policy_and_value_matrices(grid_size, start_point, end_point, obstacles):
    directions = ["↑", "↓", "←", "→"]
    policy_matrix = np.random.choice(directions, size=(grid_size, grid_size)).astype(object)

    for (i, j) in obstacles:
        policy_matrix[i][j] = '■'

    policy_matrix[start_point[0]][start_point[1]] = 'S'
    policy_matrix[end_point[0]][end_point[1]] = 'E'

    value_matrix = evaluate_policy(grid_size, policy_matrix, obstacles, start_point, end_point)

    value_grid = []
    policy_grid = []

    for i in range(grid_size):
        value_row = []
        policy_row = []
        for j in range(grid_size):
            value_row.append(value_matrix[i][j])
            policy_row.append(policy_matrix[i][j])
        value_grid.append(value_row)
        policy_grid.append(policy_row)

    return value_grid, policy_grid


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        grid_size = int(request.form.get("n"))
        start_point = tuple(map(int, request.form.get("start").split(',')))
        end_point = tuple(map(int, request.form.get("end").split(',')))
        obstacles_data = request.form.get("obstacles").split()

        obstacles = [tuple(map(int, obs.split(','))) for obs in obstacles_data]

        value_grid, policy_grid = generate_policy_and_value_matrices(grid_size, start_point, end_point, obstacles)
        return render_template("result.html", n=grid_size, value_matrix=value_grid, policy_matrix=policy_grid)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
