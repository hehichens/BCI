import argparse

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", help="data name {aa, al, av, aw ay, all, single}", \
        default="all", type=str)
    parser.add_argument("--method", help="feature method {bandpowers, dct, mrk, csp, all, single}", \
        default="single", type=str)

    args = parser.parse_args()
    return args


opt = options()