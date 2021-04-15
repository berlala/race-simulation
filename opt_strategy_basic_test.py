import numpy as np
import cvxpy as cp
import racesim_basic
import helper_funcs.src.calc_tire_degradation

race_pars_file_ = "pars_YasMarina_2017.ini"  #比赛赛道与车辆配置文件
use_print = True
pars_in = racesim_basic.src.import_pars.import_pars(use_print=use_print, race_pars_file=race_pars_file_)


tot_no_laps=pars_in['race_pars']['tot_no_laps']
tire_pars=pars_in['driver_pars']["tire_pars"]
sim_opts = {"min_no_pitstops": 1,
                "max_no_pitstops": 2,
                "start_compound": None,
                "start_age": 0,
                "enforce_diff_compounds": True,
                "use_qp": False,
                "fcy_phases": None} # 仿真配置

strategy_combinations = helper_funcs.src.get_strat_combinations. \
    get_strat_combinations(available_compounds=pars_in['available_compounds'],
                            min_no_pitstops=sim_opts["min_no_pitstops"],
                            max_no_pitstops=sim_opts["max_no_pitstops"],
                            enforce_diff_compounds=sim_opts["enforce_diff_compounds"],
                            start_compound=sim_opts["start_compound"],
                            all_orders=False)
cur_comp_strat = strategy_combinations[2] # 数字是停站次数
tires = [[comp, 0] for comp in cur_comp_strat]
tires[0][1] = sim_opts["start_age"]


# ------------------------------------------------------------------------------------------------------------------
# SET UP PROBLEM MATRICES ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# the basic idea is to have one design variable per stint and using the equality constraint to assure that the total
# number of laps of the race is considered (e.g. 2 stops means 3 design variables)
# set together tire degradation coefficients
print('x is '+str(tires))
print('Tire Para ' + str(tire_pars))
x = tires
#print('x select is '+str(x[0]))
#print(tire_pars['A3']['k_1_lin'])
k_1_lin_array = []
k_0_array = []
age_array = []

for i in range(len(x)):
    #print(i)
    #print(tire_pars[x[0][0][0]])
    k_1_lin_array.append([tire_pars[x[i][0][0]]['k_1_lin']])#注意取数的层级,中间的【0】是固定的
    k_0_array.append([tire_pars[x[i][0][0]]['k_0']])
    age_array.append([x[i][1]]) # 初始寿命系数

k_1_lin_array=np.array(k_1_lin_array)
k_0_array=np.array(k_0_array)
age_array=np.array(age_array)

print(k_1_lin_array)
#print(k_0_array)
#print(age_array)

# get number of stints
no_stints = len(tires)

# set up problem matrices (P = H and q = f in quadprog)
P = np.eye(no_stints) * 0.5 * k_1_lin_array * 2  # * 2 because of standard form
q = (0.5 + age_array) * k_1_lin_array + k_0_array
q = np.transpose(q) # 此处注意要转置，q应该是[1*n]的vector

G = np.eye(no_stints) * -1.0  # minimum 1 lap per stint
h = np.ones(no_stints) * -1.0

A = np.ones((1, no_stints))  # sum of stints must equal total number of laps
b = np.array([tot_no_laps])

# ------------------------------------------------------------------------------------------------------------------
# SET UP SOLVER SPECIFIC PROBLEM -----------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# create integer design variables
x = cp.Variable(shape=(no_stints), integer=True)

# create quadratic system matrix
P = cp.Constant(P)

print('===Dig for Opt Question===')
print(x)
print(no_stints)
print('============H============')
print(P)
print('============f============')
print(q)  # 冲突
print(cp.quad_form(x, P))
# set up problem using objective and constraints
objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q@x)
constraints = [G @ x <= h, A @ x == b]
prob = cp.Problem(objective, constraints)

# ------------------------------------------------------------------------------------------------------------------
# CALL SOLVER ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# ECOS_BB is able to handle mixed integer quadratic problems
tmp = prob.solve(solver='ECOS_BB')

if not np.isinf(tmp):
    stint_lengths = np.round(x.value).astype(np.int32)
else:
    # no solution found
    stint_lengths = None

print('Result is '+str(stint_lengths))

# tire age not considered in these equations!
# 1 stop 1 DV (no equality constraints -> more complicated, but less solving effort)
# H = np.array([[0.5 * k_11 + 0.5 * k_12]]) * 2  # * 2 because of standard form
# f = np.array([0.5 * k_11 + k_01 - k_12 * tot_no_laps - 0.5 * k_12 - k_02])
# G = np.array([[-1.0], [1.0]])
# h = np.array([-1.0, tot_no_laps])

# tire age not considered in these equations!
# 2 stop 2 DV (no equality constraints -> more complicated, but less solving effort)
# H = np.array([[0.5 * k_11 + 0.5 * k_13, 0.5 * k_13],
#               [0.5 * k_13, 0.5 * k_12 + 0.5 * k_13]]) * 2  # * 2 because of standard form
# f = np.array([0.5 * k_11 + k_01 - k_13 * tot_no_laps - 0.5 * k_13 - k_03,
#               0.5 * k_12 + k_02 - k_13 * tot_no_laps - 0.5 * k_13 - k_03])
# G = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
# h = np.array([-1.0, -1.0, tot_no_laps - 2, tot_no_laps - 2])
