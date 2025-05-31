from .ncco import NCCO

def generate_nccos(price_deltas, base_price, bit_mode, ncco_id_start=0):
    nccos = []
    for i, delta in enumerate(price_deltas):
        ncco = NCCO(id=ncco_id_start + i, price_delta=delta, base_price=base_price, bit_mode=bit_mode)
        nccos.append(ncco)
    return nccos 