def score_nccos(nccos):
    for ncco in nccos:
        # Example scoring logic: score = abs(price_delta) * bit_mode
        ncco.score = abs(ncco.price_delta) * ncco.bit_mode
    return nccos 