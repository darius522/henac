from torch import Tensor
from typing import List
from julius import resample_frac
from julius.resample import ResampleFrac
from torch.nn import ModuleDict, Sequential
from audiotools import AudioSignal

class MultibandResampler(object):
    def __init__(self, sampling_rate: int = 24_000, cutoffs:List[int] | None = None):
        self.sampling_rate = sampling_rate
        self.filters = ModuleDict()
        self.cutoffs = cutoffs
        for co in self.cutoffs:
            self.filters[str(co)] = Sequential(
                ResampleFrac(sampling_rate, co).to('cuda'),
                ResampleFrac(co, sampling_rate).to('cuda'),
            )

    def __call__(self, tensors):
        prev_sig, min_len = None, float('inf')
        signals = []
        for co in self.cutoffs:
            sig = self.filters[str(co)](tensors.audio_data)
            min_len = min(min_len, sig.shape[-1])
            subband = sig.clone()
            if prev_sig is not None:
                subband[...,:min_len] -= prev_sig[...,:min_len]
            signals.append(subband[...,:min_len])
            prev_sig = sig
            
        return signals

def resample_bands(sigs: List[Tensor], target_length: int | List[int]):
    if isinstance(target_length, int):
        return [resample_frac(sig, sig.shape[-1], target_length) for sig in sigs]
    return [resample_frac(sig, sig.shape[-1], tl) for sig, tl in zip(sigs, target_length)]
