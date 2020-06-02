import main_racesim
import os
import pickle
import numpy as np


def test_racesim():
    # user input
    race_pars_file_ = "pars_YasMarina_2017.ini"
    mcs_pars_file_ = 'pars_mcs.ini'
    sim_opts_ = {"use_random": False,
                 "no_bunches": 1,
                 "no_races_per_bunch": 1,
                 "no_workers": 1,
                 "use_print": False,
                 "use_print_result": False,
                 "use_plot": False}

    # simulation call
    result_objects = main_racesim.main(sim_opts=sim_opts_,
                                       race_pars_file=race_pars_file_,
                                       mcs_pars_file=mcs_pars_file_)

    # testing
    repo_path_ = os.path.dirname(os.path.abspath(__file__))
    target_race_path_ = os.path.join(repo_path_, ".github", "testobjects", "testobj_racesim_YasMarina_2017.pkl")

    with open(target_race_path_, 'rb') as fh:
        target_race = pickle.load(fh)

    assert np.allclose(target_race.racetimes, result_objects[0].racetimes)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    test_racesim()
