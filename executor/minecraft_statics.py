"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""

CLASSES = ["pig", "cow", "sheep", "chicken", "wolf", "horse", "villager", "treeA", "treeB", "armorstand", "boat", "minecart"]
DIRECTIONS = ["facing_forward", "facing_backward", "facing_right", "facing_left"]
NATURES = ["animal", "human", "plant", "inanimated_object"]

ATTRIBUTES_ALL = CLASSES + DIRECTIONS + NATURES

ANSWER_CANDIDATES = {
    # Count questions
    "count-all": ["0", "1", "2", "3", "4", "5", "6"],
    "count-other": ["0", "1", "2", "3", "4", "5", "6"],
    "count-all-group": ["0", "1", "2", "3", "4", "5", "6"],
    "count-attribute": ["0", "1", "2", "3", "4", "5", "6"],
    "count-attribure-group": ["0", "1", "2", "3", "4", "5", "6"],
    "count-obj-rel-imm": ["0", "1", "2", "3", "4", "5", "6"],
    "count-obj-rel-imm2": ["0", "1", "2", "3", "4", "5", "6"],
    "count-obj-rel-early": ["0", "1", "2", "3", "4", "5", "6"],
    "count-obj-exclude-imm": ["0", "1", "2", "3", "4", "5", "6"],
    "count-obj-exclude-early": ["0", "1", "2", "3", "4", "5", "6"],

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
