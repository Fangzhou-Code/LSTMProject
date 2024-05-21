import matplotlib.pyplot as plt

def get_route_coordinates(route_number):
    if route_number == 1:
        return ((0, 0), (5, 5))
    elif route_number == 2:
        return ((5, 5), (10, 0))
    elif route_number == 3:
        return ((10, 0), (0, 0))
    else:
        print("Invalid route number. Please enter a number between 1 and 3.")
        return None

def plot_routes(route_numbers):
    plt.figure(figsize=(8, 6))

    for route_number in route_numbers:
        start_point, end_point = get_route_coordinates(route_number)
        start_x, start_y = start_point
        end_x, end_y = end_point
        plt.plot([start_x, end_x], [start_y, end_y], marker='o', linestyle='-')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Car Routes')
    plt.grid(True)
    plt.show()

if __name__=='__main__':
    route_number = 1
    start_point, end_point = get_route_coordinates(route_number)
    print("Route", route_number, "coordinates:")
    print("Start point:", start_point[0], start_point[1])
    print("End point:", end_point[0],end_point[1])
    routes = [1, 2,3]
    plot_routes(routes)
