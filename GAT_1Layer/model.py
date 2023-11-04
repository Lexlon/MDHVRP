import torch
import torch.nn as nn

# import sys
# sys.path.append('../')
from dataset import generate_data
# from encoder import GraphAttentionEncoder
# from decoder import DecoderCell

from encoder import GraphAttentionEncoder
from decoder import DecoderCell


class AttentionModel(nn.Module):

    def __init__(self, embed_dim=128, n_encode_layers=3, n_heads=8, tanh_clipping=10., FF_hidden=512, gru_dim = 207, cars=9):
        super().__init__()

        self.Encoder = GraphAttentionEncoder(embed_dim, n_heads, n_encode_layers, FF_hidden)
        self.Decoder = DecoderCell(embed_dim, n_heads, tanh_clipping, gru_dim,cars)
        #print('gru_dim:', gru_dim)
        #print('cars:', cars)

    def forward(self, x, return_pi=False, decode_type='greedy'):
        encoder_output = self.Encoder(x)
        decoder_output = self.Decoder(x, encoder_output, return_pi=return_pi, decode_type=decode_type)
        if return_pi:
            cost, ll, pi = decoder_output
            return cost, ll, pi
        cost, ll = decoder_output
        return cost, ll


if __name__ == '__main__':

    model = AttentionModel()
    model = model.to('cuda:0')
    model.train()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = generate_data(device, batch=1,n_car_each_depot=3, n_depot=3, n_customer=20, seed=123)
    return_pi = False
    output_1 = model(data, decode_type='sampling', return_pi=True)
    output_2 = model(data, decode_type='greedy', return_pi=True)
    #if return_pi:
    cost1, ll1, pi1 = output_1
    cost2, ll2, pi2 = output_2
    print('\ncost: ', cost1.size(), cost1)
    print('\nll: ', ll1.size(), ll1)
    #print('\npi: ', pi1.size(), pi1)
    print('\ncost: ', cost2.size(), cost2)
    print('\nll: ', ll2.size(), ll2)
    #print('\npi: ', pi2.size(), pi2)
    '''
    else:
        print(output[0])  # cost: (batch)
        print(output[1])  # ll: (batch)

    cnt = 0
    for k, v in model.state_dict().items():
        print(k, v.size(), torch.numel(v))
        cnt += torch.numel(v)
    print('total parameters:', cnt)
    '''
# output[1].mean().backward()
# print('grad: ', model.Decoder.Wk1.weight.grad[0][0])
# https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py