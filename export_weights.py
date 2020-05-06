import pickle as pkl
import torch

if __name__ == '__main__':
    input = 'output_mid/model_final.pth'
    output = 'slimmer_rcnn.pkl'

    obj = torch.load(input, map_location='cpu')['model']

    newmodel = {}

    for k in list(obj.keys()):
        print(k)
        newmodel[k] = obj.pop(k).detach().numpy()

    new = {
        "model": newmodel,
        "__author__": "JimLee1996",
        "matching_heuristics": True
    }

    with open(output, 'wb') as f:
        pkl.dump(new, f)
