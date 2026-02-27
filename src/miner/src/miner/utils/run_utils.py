from common.models.api_models import RunInfo


# TODO: consolidate these functions into one
def identify_best_run(run_info_list: list[RunInfo], is_miner_pool: bool = False) -> RunInfo:
    # best run is based on
    # 1. authenticated = True
    # 2. highest incentive_perc reduced by the burn_factor and proportioned by the number of miners + 1 (you)

    applicable_runs = [r for r in run_info_list if r.authorized and r.is_miner_pool == is_miner_pool]
    # Fallback: if no runs match the is_miner_pool filter, try all authorized runs
    if len(applicable_runs) == 0:
        applicable_runs = [r for r in run_info_list if r.authorized]
    if len(applicable_runs) == 0:
        raise Exception("Fatal Error: No applicable runs found")

    if len(applicable_runs) == 1:
        return applicable_runs[0]

    best_run = max(applicable_runs, key=lambda x: x.incentive_perc * (1 - x.burn_factor) / (x.num_miners + 1))
    return best_run


def get_miner_pool_run(run_info_list: list[RunInfo]) -> RunInfo:
    return identify_best_run(run_info_list=run_info_list, is_miner_pool=True)
