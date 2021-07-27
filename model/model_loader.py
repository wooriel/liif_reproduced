import torch
import torch.nn as nn

from model.encoder import edsr

from model.decoder import baseline

def load_model(type, encod, decod=None):
    '''encoder'''
    encod_type = encod.get('type')
    encod_args = encod.get('args', None)
    encoder_fun = ''
    if encod_type == 'edsr':
        resblocks=16
        features=64
        res_s=1
        s=2
        no_up=False
        if encod_args is not None:
            # check for the arguments
            s = encod_args.get('scale', 1)
            no_up = encod_args.get('upsample', True)
            # directly call?
            encoder_fun = edsr.EDSR(depth=resblocks, channel=features, kernel_size=3,
                        res_scale=res_s, img_scale=s, use_upsampling=no_up)
        else:
            encoder_fun = edsr.EDSR(depth=resblocks, channel=features, kernel_size=3,
                        res_scale=res_s, img_scale=s, use_upsampling=no_up)
            # print(summary([16, 3, 48, 48], encoder_fun))
            print(str(encoder_fun))
    elif encod_type == 'rdn':
        return
        
    '''decoder'''
    if type == 'baseline':
        print(str(baseline.Baseline(encoder_fun)))
        return baseline.Baseline(encoder_fun)
    elif type == 'liif':
        return
    elif type == 'metasr':
        return
    else:
        raise Exception("Not defined type")


def read_yaml(yaml_file):
    # fname = os.path.join(path, name)
    # f = open(fname, 'r')
    # yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    model = yaml_file['model']
    ty = model['type']
    encoder = model['encoder']
    decoder = model.get('mlp_decoder', None)
    # print(encoder)
    # print(decoder)
    model = load_model(ty, encoder, decoder)
    return model
