# -*- coding: utf-8 -*-

# -- Sheet --

# **Installation of "SimPy" if not available**


#!pip install simpy

# **Import of required modules**


import simpy
import numpy as np
import numpy.random as random
import math

MAX_SIMULATION_TIME = 50000
VERBOSE = False
SOURCE_LAMBDA = 0.05                    # Lambda of the source : 3 customers per 1 hour
POPULATION = 5000000           
SERVICE_DISCIPLINE = 'FCFS'             # First Come First Serve
LOGGED = True
PLOTTED = True
NUM_QUEUE = 7
NUM_SERVER = [1, 4, 4, 4, 4, 4, 1]      # Number servers of each queue

# **There are three types of order's service:**
# 1. Only wash
# 2. Only dry
# 3. Wash & dry
# 
# **And two types of weight:**
# 1. 0kg $<$ weight $\le$ 10kg
# 2. 10kg $<$ weight $\le$ 20kg 


# Generate random probabilities for each type using a uniform distribution
random_type_probabilities = np.random.uniform(0, 1, size=3)
# Normalize to ensure the sum is 1
random_type_probabilities = np.round(random_type_probabilities/random_type_probabilities.sum(), decimals=1)  


# Generate random probabilities for each weight using a uniform distribution
random_weight_probabilities = np.random.uniform(0, 1, size=2)
# Normalize to ensure the sum is 1
random_weight_probabilities = np.round(random_weight_probabilities/random_weight_probabilities.sum(), decimals=1)  

# Access the generated probabilities for each type
prob_only_wash = random_type_probabilities[0]
prob_wash_and_dry = random_type_probabilities[1]
prob_only_dry = random_type_probabilities[2]

# Access the generated probabilities for each type
prob_mini = random_weight_probabilities[0]
prob_big = random_weight_probabilities[1]

LAMBDA = [
            SOURCE_LAMBDA, 
            SOURCE_LAMBDA*prob_mini*(prob_only_wash + prob_wash_and_dry), 
            SOURCE_LAMBDA*prob_big*(prob_only_wash + prob_wash_and_dry), 
            SOURCE_LAMBDA*prob_mini*(prob_only_dry + prob_wash_and_dry), 
            SOURCE_LAMBDA*prob_big*(prob_only_dry + prob_wash_and_dry), 
            SOURCE_LAMBDA*(prob_mini*(prob_wash_and_dry + prob_only_dry) + prob_big*(prob_wash_and_dry + prob_only_dry)), 
            SOURCE_LAMBDA
         ]                                         # customer/minute


MU = [1/5, 1/45, 1/55, 1/25, 1/30, 1/20, 1/5]      # customer/minute

# **Discrete - Event Simulation model**
# 
# **The definition of a job.**
# 
# The properties of a job are
# 1.  job execution time
# 2.  job arrival time


class Job:
    def __init__(self, name, arrivalTime, duration):
        self.name = name
        self.arrivalTime = arrivalTime
        self.duration = duration

    def __str__(self):
        return '%s at %d, length %d' % (self.name, self.arrivalTime, self.duration)

# **The definition of server**
# 
# There are two arguments needed for a server:
# 1. env: SimPy environment
# 2. queue discipline: FIFO (First In First Out)


class Server:
    def __init__(self, env, serverNum, strat='FIFO'):
        self.env = env
        self.strat = strat
        self.serverNum = serverNum                      # Number of server
        self.Jobs = []                                  # Single shared queue for all servers
        self.serverSleeping = [None] * self.serverNum   # List to store the sleeping status of each server
        ''' statistics '''
        self.waitingTime = 0
        self.serviceTime = 0
        self.idleTime = [0] * self.serverNum            # List to store idle time for each server
        self.jobsDone = [0] * self.serverNum            # List to store the number of done jobs for each server
        ''' register a new server process for each server '''
        for i in range(self.serverNum):
            env.process(self.serve(i))

    def serve(self, server_id):
        while True:
            ''' do nothing, just change server to idle
              and then yield a wait event which takes infinite time
            '''
            if len(self.Jobs) == 0:
                self.serverSleeping[server_id] = env.process(self.waiting(self.env, server_id))
                t1 = self.env.now
                yield self.serverSleeping[server_id]
                ''' accumulate the server idle time'''
                self.idleTime[server_id] += self.env.now - t1
            else:
                ''' get the first job to be served'''
                j = self.Jobs.pop(0)
                if LOGGED:
                    qlog.write('%.4f\t%d\t%d\n'
                               % (self.env.now, 1 if len(self.Jobs) > 0 else 0, len(self.Jobs)))

                ''' sum up the waiting time'''
                self.waitingTime += self.env.now - j.arrivalTime
                ''' yield an event for the job finish'''
                yield self.env.timeout(j.duration)
                ''' sum up the service time'''
                self.serviceTime += j.duration
                ''' sum up the jobs done '''
                self.jobsDone[server_id] += 1

    def waiting(self, env, server_id):
        try:
            if VERBOSE:
                print('Server %d is idle at %.2f' % (server_id, self.env.now))
            yield self.env.timeout(MAX_SIMULATION_TIME)
        except simpy.Interrupt as i:
            if VERBOSE:
                print('Server %d waken up and works at %.2f' % (server_id, self.env.now))        

# **The arrival process**
# 
# The arrival process is exponentially distributed which is parameterized by
# 
# 1.  number of servers
# 2.  maximum number of population
# 3.  arrival rate λ
# 4.  service rate μ
#     
# Note that, the implementation of the arrival process embeds both arrival and service distributions.


class JobGenerator:
    def __init__(self, env, server, nrjobs=10000, lam=5, mu=8):
        self.server = server
        self.nrjobs = nrjobs
        self.interarrivaltime = 1 / lam
        self.servicetime = 1 / mu
        env.process(self.generatejobs(env))

    def generatejobs(self, env):
        i = 1
        while True:
            '''yield an event for new job arrival'''
            job_interarrival = random.exponential(self.interarrivaltime)
            yield env.timeout(job_interarrival)

            ''' generate service time and add job to the shared queue'''
            job_duration = random.exponential(self.servicetime)
            self.server.Jobs.append(Job('Job %s' % i, env.now, job_duration))
            if VERBOSE:
                print('job %d: t = %.2f, l = %.2f, dt = %.2f'
                      % (i, env.now, job_duration, job_interarrival))
            i += 1

            ''' if any server is idle, wake one of them up'''
            for server_id in range(self.server.serverNum):
                if not self.server.serverSleeping[server_id].triggered:
                    self.server.serverSleeping[server_id].interrupt('Wake up, please.')
                    # Only one server is woken up, so once a server wake up, the loop is exited
                    break   

# **Open the log file**
# 
# If requested.


if LOGGED:
    for i in range(NUM_QUEUE):
        qlog = open('mm1-l%d-m%d.csv' % (LAMBDA[i], MU[i]), 'w')
        qlog.write('0\t0\t0\n')

# **Start SimPy environment**


env = simpy.Environment()

laundryQueue = []
laundryJobGenerator = []

for i in range(NUM_QUEUE):
    laundryQueue.append(Server(env, NUM_SERVER[i], SERVICE_DISCIPLINE))
    laundryJobGenerator.append(JobGenerator(env, laundryQueue[i], POPULATION, LAMBDA[i], MU[i]))

# **Run the simulation**


env.run(until=MAX_SIMULATION_TIME)

# **Close the log file**


if LOGGED:
    qlog.close()

# **Print some statistics**


RHO             = []
totalIdleTime   = []
totalJobsDone   = []
meanWaitingTime = []

#Probability of 0 job in the system
def calculateP0(numServer, rho):
    return (1 + (numServer * rho)**numServer / (math.factorial(numServer) * (1 - rho)) + sum((numServer * rho)**n / math.factorial(n) for n in range(1, numServer)))**(-1)

def calculateMeanWaitingTime(p0, rho, numServer, lam):
    return p0 * ((rho * (numServer * rho)**numServer) / (math.factorial(numServer) * lam * (1 - rho)**2))

NAME_OF_QUEUE = ['Check in queue', 'Mini wash queue', 'Big wash queue', 'Mini dry queue', 'Big dry queue', 'Iron', 'Summary']

for i in range(NUM_QUEUE):
    RHO.append(LAMBDA[i] / (NUM_SERVER[i] * MU[i]))
    totalIdleTime.append(sum(laundryQueue[i].idleTime))
    totalJobsDone.append(sum(laundryQueue[i].jobsDone))
    meanWaitingTime.append(calculateMeanWaitingTime(calculateP0(NUM_SERVER[i], RHO[i]), RHO[i], NUM_SERVER[i], LAMBDA[i]))
    print(NAME_OF_QUEUE[i])
    print('Arrivals               : %d' % (totalJobsDone[i]))
    print('Utilization            : %.2f/%.2f' % (1.0 - totalIdleTime[i] / (MAX_SIMULATION_TIME * NUM_SERVER[i]), RHO[i]))
    print('Mean waiting time      : %.2f/%.2f' % ( laundryQueue[i].waitingTime / totalJobsDone[i], meanWaitingTime[i]))
    print('Mean service time      : %.2f/%.2f' % (laundryQueue[i].serviceTime / totalJobsDone[i], 1/MU[i]))
    print('Mean response time     : %.2f/%.2f' % (((laundryQueue[i].waitingTime + laundryQueue[i].serviceTime )/ totalJobsDone[i]),(meanWaitingTime[i] + 1/MU[i])))
    print('------------------------------------------------------------')

