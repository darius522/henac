import os
import sys
import torch
from collections import OrderedDict
from dac.model import DAC

def run_stage1(ckpt_path):
    print(f"Running Stage 1 (CB -> MB) with checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
    state_dict = {k: v for k, v in state_dict.items() if "skip_aes" not in k}

    skip_args = dict(
        codebook_dim=4096,
        n_codebooks=2,
        encoder_rates=[3],
        decoder_rates=[3],
        quantizer_dropout=0.0,
        gumbel_softmax=False,
        diff_entropy=False,
    )

    generator = DAC(skip_args=skip_args)
    missing_keys, unexpected_keys = generator.load_state_dict(state_dict, strict=False)
    state_dict = generator.state_dict()

    keys = {'multidecoders': OrderedDict(), 'decoder': OrderedDict()}
    for k, v in state_dict.items():
        if 'skip_aes' in k:
            continue
        if 'decoder' in k and not 'decoder.model.0' in k and '.skip.' not in k:
            keys[k.split('.')[0]][k] = v

    print(len(keys['multidecoders']), len(keys['decoder']))

    v1, v2 = keys['multidecoders'], keys['decoder']
    for (kk1, vv1), (kk2, vv2) in zip(v1.items(), v2.items()):
        assert vv1.size() == vv2.size()
        keys['decoder'][kk2] = vv1

    state_dict.update(keys['decoder'])
    missing_keys, unexpected_keys = generator.load_state_dict(state_dict, strict=True)

    sig = torch.randn((1, 1, 24_000))
    out1 = generator(sig)
    multidec_audio = out1['audio']
    dec_audio = generator.decode(out1['z'])
    print("Allclose check:", torch.allclose(multidec_audio, dec_audio))

    tmp_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    tmp_dict['state_dict'] = state_dict
    save_path = ckpt_path.replace('weights.pth', 'weights_resave.pth')
    torch.save(tmp_dict, save_path)
    print(f"Resaved modified checkpoint at {save_path}")


def run_stage2(ckpt_path):
    print(f"Running Stage 2 (MB -> HB) with checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']

    skip_args = {
        0: dict(
            codebook_dim=4096,
            n_codebooks=2,
            encoder_rates=[3],
            decoder_rates=[3],
            quantizer_dropout=0.0,
            gumbel_softmax=False,
            diff_entropy=False,
        ),
        1: dict(
            codebook_dim=16384,
            n_codebooks=1,
            encoder_rates=[6],
            decoder_rates=[6],
            quantizer_dropout=0.0,
            gumbel_softmax=False,
            diff_entropy=False,
        ),
    }

    generator = DAC(skip_args=skip_args)
    missing_keys, unexpected_keys = generator.load_state_dict(state_dict, strict=False)
    state_dict = generator.state_dict()

    keys = {'multidecoders.0': OrderedDict(), 'multidecoders.1': OrderedDict()}
    for k, v in state_dict.items():
        if 'multidecoders' in k and not 'multidecoders.0.model.0' in k and '.skip.' not in k:
            keys['.'.join(k.split('.')[0:2])][k] = v

    print(len(keys['multidecoders.0']), len(keys['multidecoders.1']))

    v1, v2 = keys['multidecoders.0'], keys['multidecoders.1']
    for (kk1, vv1), (kk2, vv2) in zip(v1.items(), v2.items()):
        assert vv1.size() == vv2.size()
        keys['multidecoders.1'][kk2] = vv1

    state_dict.update(keys['multidecoders.1'])
    missing_keys, unexpected_keys = generator.load_state_dict(state_dict, strict=True)

    sig = torch.randn((1, 1, 24_000))
    feats, codes, losses = generator.encode(sig)

    xcore_blind = generator.decode(feats['core'], blind_level=0)
    xmb = generator.multidecoders[0](feats['mb'], blind_level=None, x_blind=xcore_blind)
    xmb_blind = generator.multidecoders[0](feats['mb'], blind_level=0, x_blind=xcore_blind)
    xhb_band = generator.multidecoders[1](feats['hb'], blind_level=None, x_blind=xmb_blind)

    print("Allclose check:", torch.allclose(xmb, xhb_band))

    tmp_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    tmp_dict['state_dict'] = state_dict
    save_path = ckpt_path.replace('weights.pth', 'weights_resave.pth')
    torch.save(tmp_dict, save_path)
    print(f"Resaved modified checkpoint at {save_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <stage1|stage2> <ckpt_path>")
        sys.exit(1)

    stage = sys.argv[1].lower()
    ckpt_path = sys.argv[2]

    if stage == "stage1":
        run_stage1(ckpt_path)
    elif stage == "stage2":
        run_stage2(ckpt_path)
    else:
        print("Error: first argument must be either 'stage1' or 'stage2'")
        sys.exit(1)