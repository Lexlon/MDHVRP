
import torch


from dataclass import TorchJson

from dataset import generate_data




if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # assert len(sys.argv) >= 2, 'len(sys.argv) should be >= 2'
    # print(sys.argv)
    n_depot = 3  # 3
    n_car_each_depot = 3  # 5
    n_customer = 50  # 100
    seed = 4  # 0
    #capa = 1  # 2.

    data = generate_data(device, batch=1, n_car_each_depot=n_car_each_depot, n_depot=n_depot, n_customer=n_customer,
                          seed=seed)

    basename = f'n{n_customer}d{n_depot}c{n_car_each_depot}s{seed}.json'
    dirname1 = 'data/'
    json_path_torch = dirname1 + basename

    print(f'generate {json_path_torch} ...')

    hoge1 = TorchJson(json_path_torch)
    hoge1.dump_json(data)
    data = hoge1.load_json(device)
