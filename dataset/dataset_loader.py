from torch.utils.data import DataLoader
import os
import yaml

import dataset.div2k as div2k

def load_dset(ty, dataset):
    if ty == 'div2k':
        status = dataset['status']
        # print(status)
        print("loading {} dataset".format(status))
        args = dataset['args']
        if args is not None:
            return div2k.DIV2K(status, args.get('scale', None), args.get('min_max', None))
        return div2k.DIV2K(status)
    elif ty == 'celebAHQ':
        return None
    elif ty == 'benchmark':
        return None
    else:
        raise Exception("Not supported type")

def read_yaml(yaml_file):
    # open yaml file at training time
#     fname = os.path.join(path, name)
#     f = open(fname, 'r')
#     yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    dset = yaml_file['datasets'] # get this as input later
    type = dset['type']
    # load first dataset
    d1 = dset['dataset']
    datatype1 = load_dset(type, d1)
    d2 = dset['dataset2']
    datatype2 = d2
    if d2 is not None:
        datatype2 = load_dset(type, d2)
    return datatype1, datatype2
