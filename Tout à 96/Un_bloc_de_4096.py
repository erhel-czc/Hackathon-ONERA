import numpy as np
from gnuradio import gr


class blk(gr.sync_block):

    def __init__(self):  
        gr.sync_block.__init__(
            self,
            name='Test',  
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.signal = []
        self.count = 0

    def work(self, input_items, output_items):
        self.signal = np.concatenate((self.signal,input_items[0]),axis=0)
        if len(self.signal) >= 4096 and self.count==10:
            output_items[0] = self.signal[:4096]
            self.signal = self.signal[4096:]
        else :
            output_items[0] = []
            self.count+=1
        return len(output_items[0])
