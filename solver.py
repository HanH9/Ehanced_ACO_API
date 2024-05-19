
"""Solver for CVRP problem"""
from abc import abstractmethod
from random import Random
import sys
from timeit import default_timer as timer
import numpy
import numpy as np
from matplotlib import pyplot as plt

from city import City

class BaseSolver:
    """Base class for a CVRP problem solver"""
    def __init__(self, cities: list[City], max_capacity: int, max_range: int, number_of_trucks: int, seed: int) -> None:
        self.cities = cities
        self.max_capacity = max_capacity
        self.max_range = max_range
        self.number_of_trucks = number_of_trucks
        self.distances = self._calculate_distances(cities)
        self.was_visited = [-self.number_of_trucks + 1] + [0] * (len(self.cities) - 1)
        self.rem_capacity = self.max_capacity
        self.rem_range = self.max_range
        self.current_id = 0
        self.waiting = self.cities[1:]
        self.route = []
        self.route_length = sys.maxsize
        self.result = None
        self.seed = seed
        self.random = Random(self.seed)

        # 储存按城市名形式的子路径
        self.routes_name = []
        # 储存按城市经纬度形式的子路径
        self.routes_coordinate = []

    @abstractmethod
    def solve(self, output: str = None) -> None:
        """Triggers solving the problem and outputs analytics information to output if needed"""
        raise NotImplementedError('Solving not supported in the base class, use subclass instead')

    def print_result(self) -> None:
        """Prints solver result"""
        print(self.get_algorithm_name())
        if not self.result:
            print("未找到解决方案")
        else:
            print("找到的路径")
            # 按城市名输出每条路径
            print("按城市名输出每条路径:")
            for i, path in enumerate(self._split_route(self.route)):
                city_names = [self.cities[city_id].name for city_id in path]
                print(F"{i}. {' -> '.join(city_names)}")
            print(F'总长度: {self.route_length}')

            # 按经纬度输出每条路径
            print("\n按经纬度输出每条路径:")
            for i, path in enumerate(self._split_route(self.route)):
                city_coordinates = [self.cities[city_id].coordinate for city_id in path]
                print(F"{i}. {' -> '.join(map(str, city_coordinates))}")

            print(F'总长度: {self.route_length}')

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return name of algorithm used by solver"""
        raise NotImplementedError('Algorithm name not supported in the base class, use subclass instead')

    def _can_visit(self, target_id: int) -> bool:
        range_fulfilled = ((target_id == 0 and self._get_distance_to(target_id) <= self.rem_range)
            or self._get_distance_to(target_id) + self.distances[target_id, 0] <= self.rem_range)
        return (self.current_id != target_id
                and self.was_visited[target_id] < 1
                and self.cities[target_id].demand <= self.rem_capacity
                and range_fulfilled)

    def _get_target_id(self, allowed_cities: list[int]) -> int:
        return min(allowed_cities, key=self._get_distance_to)

    def _visit(self, target_id: int, route: list[int]) -> float:
        self.was_visited[target_id] += 1
        self.rem_capacity -= self.cities[target_id].demand
        self.rem_range -= self._get_distance_to(target_id)
        distance = self._get_distance_to(target_id)
        self.current_id = target_id
        route.append(self.current_id)
        if target_id == 0:
            self.rem_capacity = self.max_capacity
            self.rem_range = self.max_range
        return distance

    def _check_all_visited(self) -> bool:
        return all(self.was_visited[1:])

    def _get_distance_to(self, target_id: int) -> float:
        return self.distances[self.current_id, target_id]

    def _find_route(self) -> tuple[list[int], float]:
        route_length = 0
        route = [0]
        while not self._check_all_visited():
            to_visit = list(filter(self._can_visit, range(0, len(self.cities))))
            if len(to_visit) == 0:
                return ([], -1)
            if 0 in to_visit and len(to_visit) > 1:
                to_visit.remove(0)
            target_id = self._get_target_id(to_visit)
            route_length += self._visit(target_id, route)
        route_length = (route_length + self._visit(0, route) if self._can_visit(0) else -1)
        return (route, route_length)

    def _update_result(self, route: list[int], route_length: float) -> None:
        if route_length < self.route_length:
            self.route = route
            self.route_length = route_length
            self.result = True

    def _split_route(self, route: list[int]) -> list[list[int]]:
        result = []

        path = [route[0]]
        for value in route[1:]:
            path.append(value)
            if value == 0:
                result.append(path)
                path = [0]
        return result

    def _get_route_length(self, route: list[int]) -> float:
        length = 0
        for i, city_from in enumerate(route[:-1]):
            city_to = route[i + 1]
            length += self.distances[city_from, city_to]
        return length

    @staticmethod
    def _calculate_distances(cities: list[City]) -> numpy.ndarray:
        distances = numpy.zeros((len(cities), len(cities)))
        for i, city1 in enumerate(cities):
            for j, city2 in enumerate(cities):
                if i > j:
                    distances[i,j] = distances[j,i] = city1.distance(city2)
        return distances

    def visualize_optimal_path(self) -> None:
        """Visualizes the optimal path by plotting each sub-path"""
        paths = self._split_route(self.route)
        colors = plt.cm.get_cmap('tab20', len(paths) + 1).colors  # 生成不同颜色
        plt.figure(figsize=(7, 5))  # 设置图像大小为7x5英寸
        for i, path in enumerate(paths):
            path_coords = [(self.cities[city_id].longitude, self.cities[city_id].latitude) for city_id in path]
            path_coords.append((self.cities[0].longitude, self.cities[0].latitude))  # 添加配送中心
            x_coords, y_coords = zip(*path_coords)

            plt.plot(x_coords, y_coords, marker='o', markersize=8, color=colors[i])
            for j, city_id in enumerate(path):
                plt.annotate(f'{city_id}', (x_coords[j], y_coords[j]), textcoords="offset points", xytext=(0,10), ha='center', fontproperties="SimSun")

        plt.plot(self.cities[0].longitude, self.cities[0].latitude, marker='*', markersize=10, color='black')
        plt.xlabel('经度', fontproperties="SimSun", size=12)
        plt.ylabel('纬度', fontproperties="SimSun", size=12)
        plt.grid(False)  # 去掉绘图背景中的标准型

        # 设置横纵坐标刻度范围
        min_longitude = min(city.longitude for city in self.cities)
        max_longitude = max(city.longitude for city in self.cities)
        min_latitude = min(city.latitude for city in self.cities)
        max_latitude = max(city.latitude for city in self.cities)
        plt.xlim(min_longitude - 0.002, max_longitude + 0.002)
        plt.ylim(min_latitude - 0.002, max_latitude + 0.002)

        plt.show()

    def store_routes_name(self) -> None:
        """Store routes by city names"""
        # Split route into subpaths
        subpaths = self._split_route(self.route)
        # Store routes by city names
        # Iterate over subpaths
        for subpath in subpaths:
            # Initialize route_name list for current subpath
            route_name = []
            # Iterate over cities in current subpath
            for city_id in subpath:
                # Get city name and append to route_name
                city_name = self.cities[city_id].name
                route_name.append(city_name)
            # Append route_name to routes_name
            self.routes_name.append(route_name)

    def store_routes_coordinate(self) -> None:
        """Store routes by city coordinates"""
        # Split route into subpaths before storing by coordinates
        subpaths = self._split_route(self.route)
        # Store routes by coordinates
        # Iterate over subpaths
        for subpath in subpaths:
            # Initialize route_coordinate list for current subpath
            route_coordinate = []
            # Iterate over cities in current subpath
            for city_id in subpath:
                # Get city coordinate and append to route_coordinate
                city_coordinate = self.cities[city_id].coordinate
                route_coordinate.append(city_coordinate)
            # Append route_coordinate to routes_coordinate
            self.routes_coordinate.append(route_coordinate)

class _AntSolution:
    def __init__(self) -> None:
        self.current_route = [0]
        self.current_route_length = -1
        self.best_route = []
        self.best_route_length = sys.maxsize

    def check_current_route(self) -> bool:
        """Checks if current solution is better than the best one and updates it if it is needed"""
        if (self.current_route_length == -1 or self.current_route_length >= self.best_route_length):
            return False
        self.best_route = self.current_route
        self.best_route_length = self.current_route_length
        return True

    def reset(self) -> None:
        """Resets current route and solution"""
        self.current_route = [0]
        self.current_route_length = -1

class ACOSolver(BaseSolver):
    """Class implementing a solver for CVRP problem using ACO"""
    def __init__(self, cities: list[City], max_capacity: int, max_range: int, number_of_trucks: int, seed: int,
                number_of_ants: int, alpha: float, beta: float, pheromones_factor: float, evaporate_factor: float, number_of_iterations: int) -> None:
        super().__init__(cities, max_capacity, max_range, number_of_trucks, seed)
        self.number_of_ants = number_of_ants
        self.alpha = alpha
        self.beta = beta
        self.pheromones_factor = pheromones_factor
        self.evaporate_factor = evaporate_factor
        self.number_of_iterations = number_of_iterations
        self.pheromones = numpy.ones((len(cities), len(cities)))
        self.current_ant_id = 0
        self.ants = [_AntSolution() for _ in range(number_of_ants)]

    def solve(self, output: str = None) -> None:
        if output is not None:
            file = open(output, 'a', encoding='utf-8')
            file.write(F'{self.get_algorithm_name()}\n')
        try:
            start = timer()
            for i in range(self.number_of_iterations):
                for ant in self.ants:
                    ant.reset()
                    self.was_visited = [-self.number_of_trucks + 1] + [0] * (len(self.cities) - 1)
                    (ant.current_route, ant.current_route_length) = self._find_route()
                    self._lay_pheromones(ant.current_route)
                    if ant.check_current_route():
                        self._update_result(ant.best_route, ant.best_route_length)
                if output is not None and not file.closed:
                    file.write(F'{i+1} {self.route_length if 0 <= self.route_length < sys.maxsize else 0}\n')
                self._update_pheromones()
            end = timer()
        finally:
            if output is not None and not file.closed:
                file.write(F'Time: {(end - start) * 1000} ms\n')
                file.close()

    def get_algorithm_name(self) -> str:
        return 'ACO'

    def _get_target_id(self, allowed_cities: list[int]) -> int:
        weights = []
        for city in allowed_cities:
            if self._get_distance_to(city) == 0:
                return city
            pheromon_factor = self.pheromones[self.current_id, city]
            heuristic_factor = 1 / self._get_distance_to(city)
            weights.append((pheromon_factor ** self.alpha) * (heuristic_factor ** self.beta))
        if sum(weights) <= 0.0:
            weights = None
        return self.random.choices(allowed_cities, weights, k = 1)[0]

    def _lay_pheromones(self, route: list[int], factor: float = None) -> None:
        if factor is None:
            factor = self.pheromones_factor
        if len(route) <= 0:
            return
        for i in range(len(route) - 1):
            city_from = route[i]
            city_to = route[i+1]
            if city_from != city_to:
                self.pheromones[city_from, city_to] += factor / self._get_route_length(route)

    def _update_pheromones(self) -> None:
        for i in range(len(self.cities)):
            for j in range(len(self.cities)):
                if i != j:
                    self.pheromones[i, j] *= (1 - self.evaporate_factor)

class EnhancedACOSolver(ACOSolver):
    """Class implementing a solver for CVRP problem using ACO with inversion of subpaths"""
    def _find_route(self) -> tuple[list[int], float]:
        (base_route, base_route_length) = super()._find_route()
        if base_route_length <= 0:
            return (base_route, base_route_length)
        return self.__reverse_subpaths(base_route)

    def get_algorithm_name(self) -> str:
        return 'Enhanced'

    def __reverse_subpaths(self, base_route: list[int]) -> tuple[list[int], float]:
        best_route, best_length = [0], 0
        paths = self._split_route(base_route)
        lengths = [self._get_route_length(path) for path in paths]
        for i, path in enumerate(paths):
            if len(path) <= 3:
                best_route.extend(paths[i][1:])
                best_length += lengths[i]
                continue
            modified_subpath = path
            for j in range(1, len(path) - 2):
                for k in range(j + 1, len(path) - 1):
                    # Reverse the subpath from j to k
                    modified_subpath[j:k + 1] = modified_subpath[j:k + 1][::-1]
                    modified_subpath_length = self._get_route_length(modified_subpath)
                    if modified_subpath_length < lengths[i]:
                        paths[i], lengths[i] = modified_subpath, modified_subpath_length
            best_route.extend(paths[i][1:])
            best_length += lengths[i]
        return (best_route, best_length)
