from ANNarchy import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import scipy.sparse
import gymnasium as gym
from scipy.special import erf
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler, CustomScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from ns_gym import base
import ns_gym.utils as utils
from typing import Union, Any, Optional,Type



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





LIF = Neuron(  #I = 75
    parameters = """
    tau = 50.0 : population
    I = 0.0
    tau_I = 10.0 : population
    """,
    equations = """
    tau * dv/dt = -v + g_exc - g_inh + (I-65) : init=0
    tau_I * dg_exc/dt = -g_exc
    tau_I * dg_inh/dt = -g_inh
    """,
    spike = "v >= -40.0",
    reset = "v = -65"
)

IZHIKEVICH = Neuron(  #I = 20
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
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
        du/dt = a*(b*v - u) : init=-14.0
    """,
    spike="v >= 30.0",
    reset="v = c; u += d"
)



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




change_state = {
    "active": False,
    "remaining": 0
}

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


I = 0
def snn(n_entrada, n_salida, n, i, matrix, inputWeights, trial, genome_id, rstdp):
    try:
        #print("rstdp: ", rstdp)
        I = i
        clear()
        pop = Population(geometry=n, neuron=IZHIKEVICH)
        #proj = Projection(pre=pop, post=pop, target='exc')
        proj = Projection(pre=pop, post=pop, target='exc', synapse=R_STDP(tau_c=rstdp[0], A_plus=rstdp[1], A_minus=rstdp[2], tau_minus=rstdp[3], tau_plus=rstdp[4]))
        #Matrix to numpy array
         # Verificar el tamaño de la matrix
        if matrix.size == 0:
            raise ValueError("matrix is empty")
        #lil_matrix scipy nxn with values of matrix
        lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))

        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        lil_matrix[:n_rows, :n_cols] = matrix
        proj.connect_from_sparse(lil_matrix)
        nombre = 'annarchy/annarchy-'+str(int(trial))+'/annarchy-'+str(int(i))
        compile(directory=nombre, clean=False, silent=True)
        M = Monitor(pop, ['spike','v'])
        input_index = []
        output_index = []
        n_entrada = int(n_entrada)
        n_salida = int(n_salida)
        for i in range(n_entrada):
            input_index.append(i)
        for i in range(n_entrada,n_salida+n_entrada):
            output_index.append(i)
        # Verificar el tamaño de inputWeights
        if inputWeights.size == 0:
            raise ValueError("inputWeights is empty")

        funcion = get_function('results/trial-'+ str(int(trial)))
        params_ns = get_params_ns('results/trial-'+ str(int(trial)))

        fit = fitness(pop,proj,M,input_index,output_index, funcion, inputWeights, genome_id*int(trial), params_ns)
        #return fit

        return fit
    except Exception as e:
        # Capturar y manejar excepciones
        print("Error en annarchy:", e)

def fitness(pop, proj ,Monitor, input_index, output_index, funcion, inputWeights, genome_id, params_ns):
    if funcion == "cartpole_ns":
        return cartpole_ns(pop, proj, Monitor, input_index, output_index, inputWeights, genome_id, params_ns)
    elif funcion == "acrobot_ns":
        return acrobot_ns(pop, proj, Monitor, input_index, output_index, inputWeights, genome_id, params_ns)
    elif funcion == "mountaincar_ns":
        return mountaincar_ns(pop, Monitor, input_index, output_index, inputWeights, genome_id, params_ns)
    elif funcion == "cartpole":
        return cartpole(pop, proj, Monitor, input_index, output_index, inputWeights, genome_id)
    elif funcion == "acrobot":
        return acrobot(pop, proj, Monitor, input_index, output_index, inputWeights, genome_id)
    else:
        raise ValueError(f"Unknown function: {funcion}")


def get_function(folder):
    # Open config file and get the parameter "function"
    config_path = folder + '/config.cfg'
    with open(config_path) as f:
        lines = f.readlines()
        for line in lines:
            if "function" in line:
                return line.split('=')[1].strip()
    return None

def get_params_ns(folder):
    config_path = folder + '/config.cfg'

    with open(config_path) as f:
        for line in f:
            if "tunable_params" in line:
                params_line = line.split('=')[1].strip()
                params_list = [p.strip() for p in params_line.split(',')]
                return params_list

    return []


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)



def cartpole_ns(pop, proj, Monitor,input_index,output_index,inputWeights, genome_id, params_ns):
    base_env = gym.make("CartPole-v1")
    scheduler = CustomScheduler(event_function)
    #scheduler2 = PeriodicScheduler(period=500)
    update_function = BoundedRandomWalk(scheduler, mu=0, sigma=10, min_val=2.0, max_val=30.0)
    update_function2 = BoundedRandomWalk(scheduler, mu=0, sigma=10, min_val=9.0, max_val=20.0)
    update_function3 = BoundedRandomWalk(scheduler, mu=0, sigma=10, min_val=0.5, max_val=0.75)
    update_function4 = BoundedRandomWalk(scheduler, mu=0, sigma=10, min_val=0.05, max_val=0.5)

    tunable_params = {}
    if "force_mag" in params_ns:
        tunable_params["force_mag"] = update_function
    if "gravity" in params_ns:
        tunable_params["gravity"] = update_function2
    if "length" in params_ns:
        tunable_params["length"] = update_function3
    if "masspole" in params_ns:
        tunable_params["masspole"] = update_function4

    #tunable_params = {"force": update_function}
    #tunable_params = {"gravity": update_function2}
    #tunable_params = {"length": update_function3}
    #tunable_params = {"masspole": update_function4}
    env = NSClassicControlWrapper(base_env, tunable_params, change_notification=True)
    observation = env.reset()[0].state
    terminated = False
    truncated = False
    #Number of episodes
    change_episode = 100
    episodes = 5
    h=0
    #Final fitness
    final_fitness = 0

    # Limits for each observation variable
    limits = [
        (-4.8, 4.8),  # Cart position
        (-10.0, 10.0),  # Cart velocity (estimated)
        (-0.418, 0.418),  # Pole angle in radians
        (-10.0, 10.0)  # Pole angular velocity (estimated)
    ]

    recompensas = []
    gravedades = []
    fuerzas = []
    largos = []
    masas = []

    while h < episodes*change_episode:
        j=0
        returns = []
        actions_done = []
        observation, info = env.reset()
        if gravedades != []:
            base_env.gravity = gravedades[-1]
            env.unwrapped.gravity = gravedades[-1]
        if fuerzas != []:
            base_env.force_mag = fuerzas[-1]
            env.unwrapped.force_mag = fuerzas[-1]
        if largos != []:
            env.unwrapped.length = largos[-1]
            env.length = largos[-1]
        if masas != []:
            env.unwrapped.masspole = masas[-1]
            env.masspole = masas[-1]
        if h % change_episode == 0:
            change_state["active"] = True
            change_state["remaining"] = change_episode

        terminated = False
        truncated = False
        while not terminated and not truncated:
            #encode observation, 4 values split in 8 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
            i = 0
            k = 0
            for val in observation.state:
                if val < 0:
                    val = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = val*30
                    pop[int(input_index[i+1])].I = 0
                else:
                    val = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*30
                i += 2
                k += 1

            simulate(50.0)
            spikes = Monitor.get('spike')
            #Output from 2 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2: #left
                action = 0
            elif output1 < output2: #right
                action = 1
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward.reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        #The fitness is the sum of the rewards for each episode
        gravedades.append(env.unwrapped.gravity)
        fuerzas.append(env.unwrapped.force_mag)
        largos.append(env.unwrapped.length)
        masas.append(env.unwrapped.masspole)
        final_fitness += np.sum(returns)
        recompensas.append(np.sum(returns))
        if len(recompensas) > 100:
            recompensas.pop(0)
        r = np.sum(returns) - np.mean(recompensas)
        proj.reward = r
        simulate(50.0)
        h += 1
        Monitor.reset()
        pop.reset()
        proj.reward = 0

    #The final fitness is the mean of the fitness for each episode
    final_fitness = final_fitness/(episodes*change_episode)
    env.close()
    return final_fitness





def acrobot_ns(pop, proj, Monitor, input_index, output_index, inputWeights, genome_id, params_ns):
    base_env = gym.make("Acrobot-v1")

    scheduler = PeriodicScheduler(period=5)
    scheduler2 = PeriodicScheduler(period=5)
    scheduler3 = PeriodicScheduler(period=5)
    update1 = BoundedRandomWalk(scheduler, min_val=0.8, max_val=2.0)
    update2 = BoundedRandomWalk(scheduler2, min_val=0.8, max_val=3.0)
    update3 = BoundedRandomWalk(scheduler3, min_val=0.2, max_val=0.8)



    tunable_params = {
        "LINK_LENGTH_1": update1,
        "LINK_LENGTH_2": update1,
        "LINK_MASS_1": update2,
        "LINK_MASS_2": update2,
        "LINK_COM_POS_1": update3,
        "LINK_COM_POS_2": update3,
    }

    env = NSClassicControlWrapper(base_env, tunable_params, change_notification=True)

    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    observation, info = env.reset()
    terminated = False
    truncated = False
    # Number of episodes
    episodes = 100
    h = 0
    # Final fitness
    final_fitness = 0


    # Definir límites para cada variable de observación
    limites = [
        (-1, 1),  # cos(theta1)
        (-1, 1),  # sin(theta1)
        (-1, 1),  # cos(theta2)
        (-1, 1),  # sin(theta2)
        (-12.5663706, 12.5663706),  # theta1_dot
        (-28.2743339, 28.2743339)  # theta2_dot
    ]
    np.random.seed(int(genome_id))

    recompensas = []
    while h < episodes:
        j = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        distancias = []
        observation = env.reset()[0]
        while not terminated and not truncated:
            # Codificar observación
            i = 0
            k = 0
            for val in observation.state:
                if val < 0:
                    #Normalizar val
                    val = normalize(val, limites[k][0], limites[k][1])
                    pop[int(input_index[i])].I = val*30
                    pop[int(input_index[i+1])].I = 0
                else:
                    #Normalizar val
                    val = normalize(val, limites[k][0], limites[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*30
                i += 2
                k += 1
            simulate(50.0)
            spikes = Monitor.get('spike')
            #Output from 3 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2 and output1 > output3:
                action = 0
            elif output2 > output1 and output2 > output3:
                action = 1
            elif output3 > output1 and output3 > output2:
                action = 2
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward.reward)
            actions_done.append(action)
            Monitor.reset()
            pop.reset()
            proj.reward = 0.0
            j += 1
        final_fitness += np.sum(returns)
        recompensas.append(np.sum(returns))
        r = (np.sum(returns) - 50)/500
        proj.reward = r
        simulate(50.0)
        h += 1
        Monitor.reset()
        pop.reset()

    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness







def mountaincar_ns(pop, proj, Monitor,input_index,output_index,inputWeights, genome_id, params_ns):
    env = gym.make("MountainCar-v0")
    observation, info = env.reset()
    terminated = False
    truncated = False
    #Number of episodes
    episodes = 10
    h=0
    #Final fitness
    final_fitness = 0

    # Limits for each observation variable
    limits = [
        (-1.2, 0.6),  # Car position
        (-0.07, 0.07),  # Car velocity (estimated)
    ]
    while h < episodes:
        j=0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        observation, info = env.reset()
        while not terminated and not truncated:
            #encode observation, 4 values split in 8 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
            i = 0
            k = 0
            for val in observation:

                if val < 0:
                    pop[int(input_index[i])].I = normalize(val, limits[k][0], limits[k][1])*30
                    pop[int(input_index[i+1])].I = 0
                else:
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = normalize(val, limits[k][0], limits[k][1])*30
                i += 2
                k += 1
            simulate(50.0)
            spikes = Monitor.get('spike')
            #Output from 3 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2:
                if output1 > output3:
                    action = 0
                else:
                    action = 2
            else:
                if output2 > output3:
                    action = 1
                else:
                    action = 2
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        #The fitness is the sum of the rewards for each episode
        final_fitness += np.sum(returns)
        h += 1
        pop.reset()
        Monitor.reset()
    #The final fitness is the mean of the fitness for each episode
    final_fitness = final_fitness/episodes
    env.close()
    return final_fitness










def cartpole(pop, proj, Monitor,input_index,output_index,inputWeights, genome_id):
    env = gym.make("CartPole-v1")

    observation = env.reset()[0]
    terminated = False
    truncated = False
    #Number of episodes

    episodes = 100
    h=0
    #Final fitness
    final_fitness = 0

    # Limits for each observation variable
    limits = [
        (-4.8, 4.8),  # Cart position
        (-10.0, 10.0),  # Cart velocity (estimated)
        (-0.418, 0.418),  # Pole angle in radians
        (-10.0, 10.0)  # Pole angular velocity (estimated)
    ]

    recompensas = []
    gravedades = []
    fuerzas = []
    largos = []
    masas = []

    while h < episodes:
        j=0
        returns = []
        actions_done = []
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            #encode observation, 4 values split in 8 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    val = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = val*30
                    pop[int(input_index[i+1])].I = 0
                else:
                    val = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*30
                i += 2
                k += 1

            simulate(50.0)
            spikes = Monitor.get('spike')
            #Output from 2 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2: #left
                action = 0
            elif output1 < output2: #right
                action = 1
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        #The fitness is the sum of the rewards for each episode
        final_fitness += np.sum(returns)
        recompensas.append(np.sum(returns))
        if len(recompensas) > 100:
            recompensas.pop(0)
        r = np.sum(returns) - np.mean(recompensas)
        proj.reward = r
        simulate(50.0)
        h += 1
        Monitor.reset()
        pop.reset()
        proj.reward = 0

    #The final fitness is the mean of the fitness for each episode
    final_fitness = final_fitness/(episodes)
    env.close()
    return final_fitness


def acrobot(pop, proj, Monitor, input_index, output_index, inputWeights, genome_id):
    env = gym.make("Acrobot-v1")



    observation, info = env.reset()
    terminated = False
    truncated = False
    # Number of episodes
    episodes = 100
    h = 0
    # Final fitness
    final_fitness = 0


    # Definir límites para cada variable de observación
    limites = [
        (-1, 1),  # cos(theta1)
        (-1, 1),  # sin(theta1)
        (-1, 1),  # cos(theta2)
        (-1, 1),  # sin(theta2)
        (-12.5663706, 12.5663706),  # theta1_dot
        (-28.2743339, 28.2743339)  # theta2_dot
    ]
    np.random.seed(int(genome_id))

    recompensas = []
    largos1 = []
    largos2 = []
    masas1 = []
    masas2 = []
    moi1 = []
    moi2 = []
    retornos_parciales = []
    change_episode = 10
    while h < episodes:
        j = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        distancias = []
        observation = env.reset()[0]
    

        while not terminated and not truncated:
            # Codificar observación
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    #Normalizar val
                    val = normalize(val, limites[k][0], limites[k][1])
                    pop[int(input_index[i])].I = val*30
                    pop[int(input_index[i+1])].I = 0
                else:
                    #Normalizar val
                    val = normalize(val, limites[k][0], limites[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*30
                i += 2
                k += 1
            simulate(50.0)
            spikes = Monitor.get('spike')
            #Output from 3 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2 and output1 > output3:
                action = 0
            elif output2 > output1 and output2 > output3:
                action = 1
            elif output3 > output1 and output3 > output2:
                action = 2
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            Monitor.reset()
            pop.reset()
            j += 1
        final_fitness += np.sum(returns)
        recompensas.append(np.sum(returns))
        retornos_parciales.append(np.sum(returns))
        if len(retornos_parciales) > 50:
            retornos_parciales.pop(0)

        r = np.sum(returns) - np.mean(retornos_parciales)

        proj.reward = r
        simulate(50.0)
        h += 1
        Monitor.reset()
        pop.reset()
        proj.reward = 0
    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness


