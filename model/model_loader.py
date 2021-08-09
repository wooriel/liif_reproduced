import torch
import torch.nn as nn

from model.encoder import edsr
from model.encoder import liif

from model.decoder import baseline
from model.decoder import mlp

def load_model(type, encod, decod, ablation):
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
            no_up = encod_args.get('no_upsample', False)
            # directly call?
            encoder_fun = edsr.EDSR(depth=resblocks, channel=features, kernel_size=3,
                        res_scale=res_s, img_scale=s, no_upsampling=no_up)
        else:
            encoder_fun = edsr.EDSR(depth=resblocks, channel=features, kernel_size=3,
                        res_scale=res_s, img_scale=s, no_upsampling=no_up)
            # print(summary([16, 3, 48, 48], encoder_fun))
            print(str(encoder_fun))
    elif encod_type == 'rdn':
        return
        
    '''decoder'''
    if type == 'baseline':
        # print(str(baseline.Baseline(encoder_fun)))
        return baseline.Baseline(encoder_fun)
    elif type == 'liif':
        # if no_fu/no_le/no_cd exist, use_fu flag will be False
        use_fu = 'no_fu' not in ablation
        use_le = 'no_le' not in ablation
        use_cd = 'no_cd' not in ablation
        cont_rep = liif.LIIF(encoder_fun, use_fu, use_le, use_cd)
        if decod:
            if decod is not True:
                hidden_lst = decod.get('hidden_lst', None)
                return mlp.MLP(cont_rep, use_le, hidden_lst)
            return mlp.MLP(cont_rep, use_le)
        return cont_rep
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
    args = model['args']
    encoder = args['encoder']
    decoder = args.get('mlp_decoder', False)
    ablation = args.get('ablation', [])
    # print(encoder)
    # print(decoder)
    model = load_model(ty, encoder, decoder, ablation)
    return model
