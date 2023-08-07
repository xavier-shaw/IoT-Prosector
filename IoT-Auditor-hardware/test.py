import numpy as np

states_center_dict = {
"power_on_muted_center": [-7.36710226e+01, 2.55747255e+00, 7.36883917e+01, 1.57897851e+00, 1.57897851e+00, -8.41108012e-07, 9.99999991e-0, 1.57897851e+00, 2.55747255e+00, -9.71717172e-02, 1.78451178e-05, 9.72638950e-02, 3.80084534e-03, 6.06060606e-04, 2.62843280e-01, 2.15476565e+00, 3.63636364e-03, 1.78451178e-05],
"power_on_unmuted_center": [-7.36710226e+01, 2.55747255e+00, 7.36883917e+01, 1.57897851e+00, 1.57897851e+00, -8.41108012e-07, 9.99999991e-01, 1.57897851e+00, 2.55747255e+00, -9.71717172e-02, 1.78451178e-05, 9.72638950e-02, 3.80084534e-03, 6.06060606e-04, 2.62843280e-01, 2.15476565e+00, 3.63636364e-03, 1.78451178e-05],
"unmuted_volume_change_center": [-7.40344480e+01, 2.00916230e+00, 7.40480274e+01, 1.38286893e+00, 1.38286892e+00, -8.27932260e-07, 9.99999997e-01, 1.38286892e+00, 2.00916230e+00, -1.50895522e-01, 5.51990050e-04, 1.52736941e-01, 2.28014131e-02, 1.70149254e-02, -1.05878445e-01, 1.94220479e+00, 3.09701493e-02, 5.51990050e-04],
"muted_volume_change_center": [-7.39741877e+01, 2.38772044e+00, 7.39903395e+01, 1.51126748e+00, 1.51126748e+00, 5.27764654e-07, 1.00000001e+00, 1.51126748e+00, 2.38772044e+00, -1.48355856e-01, 5.18130631e-04, 1.50112784e-01, 2.17580131e-02, 1.52702703e-02, -2.52736970e-02, 1.94429327e+00, 2.94932432e-02, 5.18130631e-04],
"unmuted_interaction": [-7.38926581e+01, 1.06375892e+00, 7.38998630e+01, 9.70203400e-01, 9.70203400e-01, -2.67639082e-06, 1.00000000e+00, 9.70203400e-01, 1.06375892e+00, -1.09065657e-01, 2.24326599e-05, 1.09168989e-01, 4.22403837e-03, 1.21212121e-03, -3.52938845e-01, 2.18142744e+00, 4.28030303e-03, 2.24326599e-05],
"muted_inetraction": [-7.40924580e+01, 1.29855138e+00, 7.41012347e+01, 1.08437405e+00, 1.08437405e+00, -2.49230779e-07, 9.99999998e-01, 1.08437405e+00, 1.29855138e+00, -1.04343434e-01, 1.21632997e-04, 1.05010976e-01, 5.81398864e-03, 1.59090909e-03, 9.24663561e-02, 1.87544652e+00, 6.43939394e-03, 1.21632997e-04]
}

states_radius_dict = {
    "power_on_muted_radius": 2.1207295052906514,
    "power_on_unmuted_radius": 2.145839258691756,
    "unmuted_volume_change_radius": 1.318311445295326,
    "muted_volume_change_radius": 1.5227618557257423,
    "unmuted_interaction": 2.022458699164551,
    "muted_interaction": 1.7715273640444076
}

def calculate_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))  # Euclidean distance

def compute_pairwise_distance():
    center_distances = []
    view_states = []
    for [state, center] in states_center_dict.items():
        view_states.append(state)
        for [other_state, other_center] in states_center_dict.items():
            if other_state not in view_states:
                distance = calculate_distance(center, other_center)
                center_distances.append(distance)
                print(state + " -> " + other_state + ": " + str(distance))

    return np.min(center_distances)

min_distance = compute_pairwise_distance()
print(min_distance)



 