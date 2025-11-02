

# In[1]:


import os, argbind
from dac.model import DAC
import torch
from collections import OrderedDict



# ## 1st Stage CB -> MB

# In[ ]:


ckpt_path = '/N/slate/daripete/jstsp-dac/runs_32khz/baseline_21cb_small_fr_80/300k/dac/weights.pth'
state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
state_dict = {k: v for k, v in state_dict.items() if not "skip_aes" in k}
# Skip-specific
skip_args = dict(
    codebook_dim=4096,
    n_codebooks=2,
    encoder_rates=[4],
    decoder_rates=[4],
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
    if 'decoder' in k and not 'decoder.model.0' in k and not '.skip.' in k: # we don't care about skip_ae not the first layer of the decoder
        keys[k.split('.')[0]][k] = v
print(len(keys['multidecoders']), len(keys['decoder']))


# In[ ]:


v1, v2 = keys['multidecoders'], keys['decoder']
for (kk1, vv1), (kk2, vv2) in zip(v1.items(), v2.items()):
    assert vv1.size() == vv2.size()
    keys['decoder'][kk2] = vv1  # update with multidecoder value


# In[ ]:


state_dict.update(keys['decoder']) # update decoder in og state_dict


# In[ ]:


# finally sanity check the output of both decoder, they shoudl be the same
missing_keys, unexpected_keys = generator.load_state_dict(state_dict, strict=True)


# In[ ]:


sig = torch.randn((1, 1, 32_000))
out1 = generator(sig)
multidec_audio = out1['audio']
dec_audio = generator.decode(out1['z'])
torch.allclose(multidec_audio, dec_audio)


# In[ ]:


# resave
tmp_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
tmp_dict['state_dict'] = state_dict
torch.save(tmp_dict, ckpt_path.replace('weights.pth', 'weights_resave.pth'))


# ## 2nd Stage (MB -> HB)

# In[ ]:


# ckpt_path = '/N/slate/daripete/jstsp-dac/runs2/mb_24_18cb/300k/dac/weights.pth'
# state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
# # Skip-specific
# skip_args = {
#   0: dict(
#     codebook_dim=4096,
#     n_codebooks=2,
#     encoder_rates=[3],
#     decoder_rates=[3],
#     quantizer_dropout=0.0,
#     gumbel_softmax=False,
#     diff_entropy=False,
#   ),
#   1: dict(
#     codebook_dim=16384,
#     n_codebooks=1,
#     encoder_rates=[6],
#     decoder_rates=[6],
#     quantizer_dropout=0.0,
#     gumbel_softmax=False,
#     diff_entropy=False,
#   )
# }
# generator = DAC(skip_args=skip_args)
# missing_keys, unexpected_keys = generator.load_state_dict(state_dict, strict=False)
# state_dict = generator.state_dict()

# keys = {'multidecoders.0': OrderedDict(), 'multidecoders.1': OrderedDict()}
# for k, v in state_dict.items():
#     # print(k)
#     if 'multidecoders' in k and not 'multidecoders.0.model.0' in k and not '.skip.' in k: # we don't care about skip_ae not the first layer of the decoder
#         keys['.'.join(k.split('.')[0:2])][k] = v

# print(len(keys['multidecoders.0']), len(keys['multidecoders.1']))


# # In[3]:


# v1, v2 = keys['multidecoders.0'], keys['multidecoders.1']
# for (kk1, vv1), (kk2, vv2) in zip(v1.items(), v2.items()):
#     assert vv1.size() == vv2.size()
#     keys['multidecoders.1'][kk2] = vv1  # update with multidecoder value


# # In[4]:


# state_dict.update(keys['multidecoders.1']) # update decoder in og state_dict
# # finally sanity check the output of both decoder, they shoudl be the same
# missing_keys, unexpected_keys = generator.load_state_dict(state_dict, strict=True)


# # In[5]:


# sig = torch.randn((1, 1, 24_000))

# feats, codes, losses = generator.encode(sig)

# xcore_blind = generator.decode(feats['core'], blind_level=0)
# xmb = generator.multidecoders[0](feats['mb'], blind_level=None, x_blind=xcore_blind)
# xmb_blind = generator.multidecoders[0](feats['mb'], blind_level=0, x_blind=xcore_blind)
# xhb_band= generator.multidecoders[1](feats['hb'], blind_level=None, x_blind=xmb_blind)

# torch.allclose(xmb, xhb_band)


# # In[6]:


# tmp_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
# tmp_dict['state_dict'] = state_dict
# torch.save(tmp_dict, ckpt_path.replace('weights.pth', 'weights_resave.pth'))


# # In[ ]:




