import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_route_coordinates(route_number):
    # [1, 2, 3] 小车路线
    if route_number == 1:
        return ((0, 0), (15, 15))
    elif route_number == 2:
        return ((15, 15), (10, 0))
    elif route_number == 3:
        return ((10, 0), (0, 0))
    # [4, 5, 6, 7, 8] 叉车路线
    elif route_number == 4:
        return ((0, 0), (2, 4))
    elif route_number == 5:
        return ((2, 4), (5, 6))
    elif route_number == 6:
        return ((5, 6), (10, 4))
    elif route_number == 7:
        return ((10, 4), (12, 0))
    elif route_number == 8:
        return ((12, 0), (0, 0))
    # [9, 10, 11, 12] 无人机路线
    elif route_number == 9:
        return ((0, 0, 0), (4, -3, 6))
    elif route_number == 10:
        return ((4, -3, 6), (-4, 3, 6))
    elif route_number == 11:
        return ((-4, 3, 6), (-2, 5, 4))
    elif route_number == 12:
        return ((-2, 5, 4), (0, 0, 0)) 
    else:
        print("Invalid route number. Please enter a number between 1 and 12.")
        return None

def plot_routes_3d(route_numbers):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')  # 创建3D轴

    for route_number in route_numbers:
        start_point, end_point = get_route_coordinates(route_number)
        start_x, start_y, start_z = start_point
        end_x, end_y, end_z = end_point
        ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], marker='o', linestyle='-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D UAV Routes')
    plt.grid(True)
    plt.show()


def plot_routes(route_numbers, str):
    plt.figure(figsize=(8, 6))

    for route_number in route_numbers:
        start_point, end_point = get_route_coordinates(route_number)
        start_x, start_y = start_point
        end_x, end_y = end_point
        plt.plot([start_x, end_x], [start_y, end_y], marker='o', linestyle='-')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{str} Routes')
    plt.grid(True)
    plt.show()

if __name__=='__main__':
    # 画小车路线图
    routes = [1,2,3]
    plot_routes(routes, "Car")
    # 画叉车路线图
    routes = [4,5,6,7,8]
    plot_routes(routes, "ForkLift")
    # 画UAV图
    routes = [9, 10, 11, 12]
    plot_routes_3d(routes)
