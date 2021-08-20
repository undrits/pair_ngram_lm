#!/usr/bin/env python

import os
import time

import pynini


def main(directory: str) -> None:

    files = [file for file in os.listdir(directory) if "eng_rus" in file]

    for file in files:
        eng_rus_fst = pynini.Fst.read(directory + file)
        rus_eng_fst = pynini.invert(eng_rus_fst)
        rus_eng_name = "rus_eng" + file[-5:]
        rus_eng_fst.write(directory + rus_eng_name)
        print(f"{file} inverted and saved")


if __name__ == '__main__':
    directory = "data/lms/"
    main(directory)
