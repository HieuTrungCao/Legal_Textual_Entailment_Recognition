import json
import os
import argparse



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--p_mask", default= 0.1, type=float)