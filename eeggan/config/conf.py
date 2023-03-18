import json
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    n_critic: int
    rampup: int
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
    scheduler: bool

@dataclass
class Paths:
    data_path: str
    model_path: str
    model_name: str



@dataclass
class RunConfig:
    config: Config
    paths: Paths


def load_json(path:str) ->dict:
    f = open(path)
    json_config = json.load(f)
    f.close()
    return json_config

def get_run_config() ->RunConfig:
    config_json = load_json("../config/config.json")
    paths_json = load_json("../config/paths.json")
    run_config = RunConfig
    run_config.config = Config(**config_json)
    run_config.paths = Paths(**paths_json)
    return run_config


if __name__ == '__main__':
    run_config = get_run_config()
    print(run_config.config)
