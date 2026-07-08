from ANNarchy import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler, CustomScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from ns_gym import base
import ns_gym.utils as utils
from typing import Union, Any, Optional,Type
import random as rd
import optuna

IZHIKEVICH = Neuron(
    parameters="""
        a = 0.02 : population
        b = 0.2 : population
        c = -65.0 : population
        d = 8.0 : population
        I = 0.0
        tau_I = 10.0 : population
    """,
    equations="""
        dv/dt = 0.04*v*v + 5*v + 140 - u + I + g_exc - g_inh : init=-65
        du/dt = a*(b*v - u) : init=-14.0
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
    """,
    spike="v >= 30.0",
    reset="v = c; u += d"
)



class R_STDP(Synapse):
    """
    R-STDP con trazas pre y post, y modulación por recompensa.
    La actualización del peso depende de la coincidencia temporal (STDP) y el refuerzo externo.
    """

    _instantiated = []

    def __init__(self, tau_c=20.0, a=0.1,
                 A_plus=0.01, A_minus=0.01,
                 tau_plus=20.0, tau_minus=20.0,
                 w_min=0.0, w_max=1.0):

        parameters = """
            tau_c = %(tau_c)s : projection
            a = %(a)s : projection
            A_plus = %(A_plus)s : projection
            A_minus = %(A_minus)s : projection
            tau_plus = %(tau_plus)s : projection
            tau_minus = %(tau_minus)s : projection
            w_min = %(w_min)s : projection
            w_max = %(w_max)s : projection
            reward = 0.0 : projection
        """ % locals()

        equations = """
            tau_c * dc/dt = -c : event-driven
            tau_plus  * dx/dt = -x : event-driven
            tau_minus * dy/dt = -y : event-driven
        """

        pre_spike = """
            g_target += w
            x += A_plus
            c += y
            w = w + (c * reward)
        """

        post_spike = """
            y -= A_minus
            c += x
            w = w + (c * reward)
        """


        Synapse.__init__(self,
                         parameters=parameters,
                         equations=equations,
                         pre_spike=pre_spike,
                         post_spike=post_spike,
                         name="R-STDP")

        self._instantiated.append(True)



class BoundedRandomWalk(base.UpdateFn):
    def __init__(self, scheduler: Type[base.Scheduler], mu: float = 0, sigma: float = 1,
                 min_val: Optional[float] = None, max_val: Optional[float] = None, seed=None):
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.min_val = min_val
        self.max_val = max_val
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, param: float, t: float) -> tuple[float, bool]:
        return super().__call__(param, t)

    def update(self, param: float, t: float) -> float:
        #get random value between min val and max val
        updated_param = rd.uniform(self.min_val,self.max_val)
        while updated_param == param:
            updated_param = rd.uniform(self.min_val,self.max_val)
        return updated_param


def event_function(t):
    # Si está activo, consumir duración
    if change_state["active"]:
        if change_state["remaining"] > 0:
            change_state["remaining"] -= 1
            return True
        else:
            change_state["active"] = False
            return False
    return False


change_state = {
    "active": False,
    "remaining": 0
}


def get_params_ns():
    config_path = 'config.cfg'

    with open(config_path) as f:
        for line in f:
            if "tunable_params" in line:
                params_line = line.split('=')[1].strip()
                params_list = [p.strip() for p in params_line.split(',')]
                return params_list

    return []



#1;9;43.5712
#1;10;42.4252
#2;9;-15.8397
#2;10;0.221279
#3;9;49.8525
#3;10;-15.7113
#4;9;55.313
#4;10;45.7265
#5;9;-4.33209
#5;10;3.1428
#6;9;-7.02096
#6;10;67.7384
#7;9;53.6828
#7;10;-20.626
#8;9;29.6074
#8;10;73.9112
#7;3;-45.3084

file_matrix = open("best0.txt", "r")
lines = file_matrix.readlines()
# Convertir la lista de listas en un array de numpy

max_node = 0
for line in lines:
    con = line.split(";")
    in_node = int(con[0]) -1
    out_node = int(con[1]) -1
    weight = float(con[2])
    if in_node > max_node:
        max_node = in_node
    if out_node > max_node:
        max_node = out_node
# Crear una matriz de ceros de tamaño max_node x max_node
matrix = np.zeros((max_node+1, max_node+1))
# Llenar la matriz con los pesos
for line in lines:
    con = line.split(";")
    in_node = int(con[0]) -1
    out_node = int(con[1]) -1
    weight = float(con[2])
    matrix[in_node][out_node] = weight

input_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
output_index = [12, 13,14]




print("Pesos de INICIO: ", matrix)





# Etorno base
base_env = gym.make("Acrobot-v1")
scheduler = CustomScheduler(event_function=event_function)
scheduler2 = CustomScheduler(event_function=event_function)
scheduler3 = CustomScheduler(event_function=event_function)
scheduler4 = CustomScheduler(event_function=event_function)
scheduler5 = CustomScheduler(event_function=event_function)
scheduler6 = CustomScheduler(event_function=event_function)
update1 = BoundedRandomWalk(scheduler, min_val=0.8, max_val=2.0)
update2 = BoundedRandomWalk(scheduler2, min_val=0.8, max_val=3.0)
update3 = BoundedRandomWalk(scheduler3, min_val=0.2, max_val=0.8)
update4 = BoundedRandomWalk(scheduler4, min_val=0.8, max_val=2.0)
update5 = BoundedRandomWalk(scheduler5, min_val=0.8, max_val=3.0)
update6 = BoundedRandomWalk(scheduler6, min_val=0.2, max_val=0.8)
#tunable_params = {
#"LINK_LENGTH_1": update1,
#"LINK_LENGTH_2": update4,
#"LINK_MASS_1": update2,
#"LINK_MASS_2": update5,
#"LINK_COM_POS_1": update3,
#"LINK_COM_POS_2": update6
#}
tunable_params = {}
params_ns = get_params_ns()

config_file = open("config.cfg","r")
if "length" in params_ns:
    tunable_params["LINK_LENGTH_1"] = update1
    tunable_params["LINK_LENGTH_2"] = update4
if "mass" in params_ns:
    tunable_params["LINK_MASS_1"] = update2
    tunable_params["LINK_MASS_1"] = update5
if "moi" in params_ns:
    tunable_params["LINK_COM_POS_1"] = update3
    tunable_params["LINK_COM_POS_2"] = update6

env = NSClassicControlWrapper(base_env, tunable_params, change_notification=True)

i = 0
limites = [
    (-1, 1), (-1, 1), (-1, 1), (-1, 1),
    (-12.5663706, 12.5663706), (-28.2743339, 28.2743339)
]

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)




trials = 33
from scipy import sparse
matrix = sparse.csr_matrix(matrix)

print("Pesos de INICIO: ", matrix)


retornos2 = []
total_return2 = []

tau_c=26.5768
a_minus=0.0320527
a_plus=0.0183781
tau_minus=32.5565
tau_plus=14.9765

for trial in range(trials):
    #convertir en sparse matrix

    pop = Population(max_node+1, IZHIKEVICH)

    #syn = Projection(pop, pop, target='exc')
    syn = Projection(pop, pop, target='exc', synapse=R_STDP(tau_c=tau_c, A_plus=a_plus, A_minus=a_minus, tau_plus=tau_plus, tau_minus=tau_minus))


    syn.connect_from_sparse(matrix)


    compile(directory="ns-acrobot")
    j = 0
    returns = []
    actions_done = []
    terminated = False
    truncated = False
    observation = env.reset()[0].state
    #print(observation)

    #print("Pesos de entrada: ", inputWeights)
    #print(observation)

    retornos = []


    M = Monitor(pop, ['spike','v'])
    i = 0
    acciones = []
    episodes = 10
    change_episode = 100
    total_return = 0.0
    largos1 = []
    largos2 = []
    masas1 = []
    masas2 = []
    moi1 = []
    moi2 = []
    retornos_parciales = []
    for ep in range(episodes*change_episode):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0
        #print("Comienzo del episodio %d" % (ep + 1))
        #for i, dend in enumerate(syn.dendrites):
         #   print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")
        #Prediccion de recompensa en base a la media movil de la recompensa
        distancias = []
        acciones2  = []
        returns = []
        observation = env.reset()[0]
        if largos1 != [] and largos2 != []:
            env.LINK_LENGTH_1 = largos1[-1]
            env.LINK_LENGTH_2 = largos2[-1]
            env.unwrapped.LINK_LENGTH_1 = largos1[-1]
            env.unwrapped.LINK_LENGTH_2 = largos2[-1]
        if masas1 != [] and masas2 != []:
            env.LINK_MASS_1 = masas1[-1]
            env.LINK_MASS_2 = masas2[-1]
            env.unwrapped.LINK_MASS_1 = masas1[-1]
            env.unwrapped.LINK_MASS_2 = masas2[-1]
        if moi1 != [] and moi2 != []:
            env.LINK_COM_POS_1 = moi1[-1]
            env.LINK_COM_POS_2 = moi2[-1]
            env.unwrapped.LINK_COM_POS_1 = moi1[-1]
            env.unwrapped.LINK_COM_POS_2 = moi2[-1]
        if ep % change_episode == 0:
            change_state["active"] = True
            change_state["remaining"] = change_episode
        
        while not terminated and not truncated:
            # Codificar observación
            i = 0
            k = 0
            for val in observation.state:
                if val < 0:
                    #Normalizar val
                    val = normalize(val, limites[k][0], limites[k][1])
                    pop[int(input_index[i])].I = val*30 #inputWeights[k]
                    pop[int(input_index[i+1])].I = 0
                else:
                    #Normalizar val
                    val = normalize(val, limites[k][0], limites[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*30 #inputWeights[k]
                i += 2
                k += 1
            simulate(50.0)
            #print("Recompensa: ", syn.reward)
            weights = syn.w[0]
            #print("Pesos de salida: ", weights)
            spikes = M.get('spike')
            #Output from 2vneurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            action = env.action_space.sample()
            if output1 > output2 and output1 > output3:
                action = 0
            elif output2 > output1 and output2 > output3:
                action = 1
            elif output3 > output1 and output3 > output2:
                action = 2
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward.reward
            #La recomensa será la distancia de la vara desde el punto optimo cuando esta a 90ª
            # reward = 90º - abs(observation[2])
            #print(np.rad2deg(observation[2]))
            #print(observation[2])




            M.reset()
            pop.reset()
        acciones.append(acciones2)

        #print("Episode %d reward: %f" % (ep + 1, episode_return))
        retornos.append(episode_return)
        largos1.append(env.unwrapped.LINK_LENGTH_1)
        largos2.append(env.unwrapped.LINK_LENGTH_2)
        masas1.append(env.unwrapped.LINK_MASS_1)
        masas2.append(env.unwrapped.LINK_MASS_2)
        moi1.append(env.unwrapped.LINK_COM_POS_1)
        moi2.append(env.unwrapped.LINK_COM_POS_2)
        #print("Recompensas: ", distancias)
        #print("Acciones: ", rs)
        total_return += episode_return
        retornos_parciales.append(episode_return)
        if len(retornos_parciales) > 100:
            retornos_parciales.pop(0)

        r = episode_return - np.mean(retornos_parciales)

        syn.reward = r
        #simulate(50.0)
        M.reset()
        pop.reset()
        syn.reward = 0.0
        #syn.reset(synapses=True)
    retornos2.append(retornos)
    total_return2.append(total_return / (episodes*change_episode))
    print(total_return2)
    clear()
env.close()






nombre = "base-force-rstdp"
np.save(f"retornos2_{nombre}.npy", retornos2)

np.save("largos1.npy",largos1)
np.save("largos2.npy",largos2)
np.save("masas1.npy",masas1)
np.save("masas2.npy",masas2)
np.save("moi1.npy",moi1)
np.save("moi2.npy",moi2)

promedios_trials = np.mean(retornos2, axis=1)

plt.figure(figsize=(8, 5))
plt.boxplot(promedios_trials, vert=True, patch_artist=True)
plt.title(f'Distribución de la recompensa promedio por trial - {nombre}')
plt.ylabel('Recompensa promedio')
plt.xticks([1], [nombre])
plt.tight_layout()
plt.savefig(f'boxplot_{nombre}.png')
plt.close()

mean = np.mean(retornos2, axis=0)
q25 = np.percentile(retornos2, 25, axis=0)
q75 = np.percentile(retornos2, 75, axis=0)
median = np.median(retornos2, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(mean, label='Media', color='blue')
plt.fill_between(range(len(mean)), q25, q75, color='blue', alpha=0.2, label='IQR')
plt.plot(median, label='Mediana', color='red')
plt.title(f'Convergencia de la recompensa media por episodio - {nombre}')
plt.xlabel('Episodio')
plt.ylabel('Recompensa media')
plt.legend()
plt.tight_layout()
plt.savefig(f'convergencia_{nombre}.png')
plt.close()

