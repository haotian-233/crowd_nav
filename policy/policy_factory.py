from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.lstm_ga3c import LstmGA3C
from crowd_nav.policy.lstm_ga3c_t import LstmGA3C_t
from crowd_nav.policy.orca_discrete import ORCA_discrete

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['lstm_ga3c'] = LstmGA3C
policy_factory['lstm_ga3c_t'] = LstmGA3C_t
policy_factory['orca_discrete'] = ORCA_discrete
