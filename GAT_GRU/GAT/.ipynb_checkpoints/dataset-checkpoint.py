from torch.utils.data.dataset import Dataset
CAPACITIES = {20: 4, 50: 8, 80: 13, 100: 15, 120: 18}
max_demand = 9
import torch


def generate_data(device, batch=10, n_car_each_depot=3, n_depot=3, n_customer=50, seed=123):
    if seed is not None:
        torch.manual_seed(seed)
    n_node = n_depot + n_customer
    n_car = n_car_each_depot * n_depot
    return {'depot_xy': 10 * torch.rand((batch, n_depot, 2), device=device)
        , 'customer_xy': 10 * torch.rand((batch, n_customer, 2), device=device)
        , 'demand': torch.ones(size=(batch, n_customer), device=device)
        , 'car_start_node': torch.arange(n_depot, device=device)[None, :].repeat(batch, n_car_each_depot)
        , 'car_capacity': CAPACITIES[n_customer] * torch.ones((batch, n_car), device=device)
        , 'car_level': torch.arange(n_car_each_depot, device=device)[None, :].repeat(batch, n_depot).sort()[0]
        , 'demand_level': torch.randint(low=0, high=n_car_each_depot, size=(batch, n_customer), device=device)
            }


class Generator(Dataset):
    """ https://github.com/utkuozbulak/pytorch-custom-dataset-examples
         https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
         https://github.com/nperlmut31/Vehicle-Routing-Problem/blob/master/dataloader.py
    """

    def __init__(self, device, n_samples=5120, n_car_each_depot=3, n_depot=3, n_customer=50, seed=None):
        if seed is not None:
            self.data = generate_data(device, n_samples, n_car_each_depot, n_depot, n_customer, seed)
        self.data = generate_data(device, n_samples, n_car_each_depot, n_depot, n_customer, seed)

    def __getitem__(self, idx):
        dic = {}
        for k, v in self.data.items():
            dic[k] = v[idx]
        return dic

    def __len__(self):
        return self.data['depot_xy'].size(0)


if __name__ == '__main__':
    import torch

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch, batch_steps, n_customer = 128, 10, 20
    dataset = Generator(device, n_samples=3,
                        n_car_each_depot=3, n_depot=3, n_customer=20)
    data = next(iter(dataset))
    print(data)
    print(data['car_start_node'].unsqueeze(-1))
