from .srf import srf_procedure


def test_srf():
    rankings = [
        ["A"],
        1,
        ["B"],
        ["C"],
        3,
        ["D"],
    ]
    Z = 8
    ranks = srf_procedure(rankings, Z)
    print(ranks)

    rankings_alt = [
        ["g4"],
        1,
        ["g2", "g3", "g7"],
        3,
        ["g5"],
        0,
        ["g6", "g8"],
        2,
        ["g1"],
    ]
    Z_alt = 10
    ranks = srf_procedure(rankings_alt, Z_alt)
    print(ranks)
