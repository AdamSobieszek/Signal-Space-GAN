import json
from dataclasses import dataclass
from typing import List


@dataclass
class config:
    n_critic: int
    ramup: int
    seed: int
    block_epochs: List[int]
    batch_block_list: List[int]
    l_r: float
    n_blocks: int
    n_chans: int
    n_z: int
    in_filters: int
    out_filters: int
    factor: int
    num_map_layer: int
    n_reg: int
    i_block_tmp: int
    i_epoch_tmp: int
    fade_alpha: float


def get_config():
    f = open('config.json')
    json_config = json.load(f)
    run_config = config(**json_config)
    return run_config


if __name__ == '__main__':
    run_config = get_config()
    print(run_config)
