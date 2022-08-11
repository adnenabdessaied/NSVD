"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""

COLORS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
MATERIALS = ["rubber", "metal"]
SHAPES = ["cube", "cylinder", "sphere"]
SIZES = ["large", "small"]

ATTRIBUTES_ALL = COLORS + MATERIALS + SHAPES + SIZES

ANSWER_CANDIDATES = {
    # Count questions
    "count-all": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "count-other": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "count-all-group": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "count-attribute": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "count-attribure-group": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "count-obj-rel-imm": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "count-obj-rel-imm2": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "count-obj-rel-early": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "count-obj-exclude-imm": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "count-obj-exclude-early": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],

    # Existence questions
    "exist-other": ["yes", "no"],
    "exist-attribute": ["yes", "no"],
    "exist-attribute-group": ["yes", "no"],
    "exist-obj-rel-imm": ["yes", "no"],
    "exist-obj-rel-imm2": ["yes", "no"],
    "exist-obj-rel-early": ["yes", "no"],
    "exist-obj-exclude-imm": ["yes", "no"],
    "exist-obj-exclude-early": ["yes", "no"],

    # Seek questions
    "seek-attr-imm": ATTRIBUTES_ALL,
    "seek-attr-imm2": ATTRIBUTES_ALL,
    "seek-attr-early": ATTRIBUTES_ALL,
    "seek-attr-sim-early": ATTRIBUTES_ALL,
    "seek-attr-rel-imm": ATTRIBUTES_ALL,
    "seek-attr-rel-early": ATTRIBUTES_ALL,
}


