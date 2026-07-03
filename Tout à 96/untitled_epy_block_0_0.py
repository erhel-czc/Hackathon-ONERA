import numpy as np
from gnuradio import gr


class blk(gr.sync_block):

    def __init__(self):  
        gr.sync_block.__init__(
            self,
            name='Tiago Print Block',  
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

    def work(self, input_items, output_items):
        output_items[0][:] = input_items[0]
        print(len(output_items[0]))
        print(output_items[0])
        return len(output_items[0])
