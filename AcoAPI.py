from flask import Flask, request, jsonify
from solver import EnhancedACOSolver  # 导入你的算法类
from testset_parser import CVRPTestParser2, CVRPTestParser3  # 导入测试数据解析器

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize_route():
    # 从请求中获取测试数据
    data = request.json
    # 解析测试数据
    # test_data = CVRPTestParser2.parse(data['file_path'])  # 假设请求中包含文件路径
    test_data = CVRPTestParser3.parse(data['garbage_collection_points'], data['garbage_collection_points_longitude'], data['garbage_collection_points_latitude'], data['garbage_amounts'], data['truck_capacity'], data['transfer_station_name'], data['transfer_station_longitude'], data['transfer_station_latitude'])
    # 使用你的算法求解路径
    solver = EnhancedACOSolver(test_data.cities, test_data.capacity, 30000, 32, seed=None,
                                number_of_ants=len(test_data.cities), alpha=1.5, beta=5.0, pheromones_factor=20.0, evaporate_factor=0.3, number_of_iterations=200)
    solver.solve()
    solver.store_routes_name()
    solver.store_routes_coordinate()
    # 输出结果
    result = {
        'routes_name': solver.routes_name,
        'routes_coordinate': solver.routes_coordinate,
        'route_length': solver.route_length
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
