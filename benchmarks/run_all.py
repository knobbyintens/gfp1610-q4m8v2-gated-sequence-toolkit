"""Shared benchmark harness for the gated sequence toolkit."""

from gst.layers import SGRKBlock, GatedConvBlock


BLOCKS = {"sgrk": SGRKBlock, "gated_conv": GatedConvBlock}


def run(name, dim=64):
    return BLOCKS[name](dim)


if __name__ == "__main__":
    for name in BLOCKS:
        print(name, run(name))
