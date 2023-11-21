from enum import Enum, auto
import math
import numpy as np
import random as rnd
import simpy as sp
import logging

# SIMULATION CONFIG
MAX_SIMULATION_TIME = 1000
# should use 3 log level (DEBUG, INFO, WARNING)
# DEBUG print the most; INFO print only finished job, WARNING should not print much
LOGGING_LEVEL = logging.WARNING        
LOG_FILE = None     # log file name, left None if you don't want log file
PLOTTED = False     # no plotting for now
rnd.seed(42)
np.random.seed(42)

# Create a custom logger, output to console and file (if set)
log = logging.getLogger(__name__)
log.setLevel(LOGGING_LEVEL)
# Create handlers for console (print)
console_handler = logging.StreamHandler()
console_handler.setLevel(LOGGING_LEVEL)
console_handler.setFormatter(logging.Formatter())      # use default formatter
log.addHandler(console_handler)
if LOG_FILE is not None:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(LOGGING_LEVEL)
    file_handler.setFormatter(logging.Formatter())      # use default formatter
    log.addHandler(file_handler)

# SYSTEM PARAMETERS
## SOURCE PARAMETERS
SOURCE_LAMBDA = 30
POPULATION = 5000000

# Each service in the system (checkin, wash, dry, iron/folding, checkout)
# will be represented as a Server object (queue included), we config each 
# service below.
#
# All rate is measure per minute.
#
# SERVICE CONFIG
## CHECKIN
CHECKIN_QUEUE_LENGTH = math.inf
CHECKIN_SERVER_NUM = 1                  # 1 server only (arbitrary)
CHECKIN_MEAN_SERVICE_RATE = 50          # is average mu of each server
## BIGWASH
BIGWASH_QUEUE_LENGTH = math.inf
BIGWASH_SERVER_NUM = 4                  # 1 server only (arbitrary)
BIGWASH_MEAN_SERVICE_RATE = 20          # is average mu of each server
## SMALLWASH
SMALLWASH_QUEUE_LENGTH = math.inf
SMALLWASH_SERVER_NUM = 4                # 1 server only (arbitrary)
SMALLWASH_MEAN_SERVICE_RATE = 20        # is average mu of each server
## BIGDRY
BIGDRY_QUEUE_LENGTH = math.inf
BIGDRY_SERVER_NUM = 4                   # 1 server only (arbitrary)
BIGDRY_MEAN_SERVICE_RATE = 20           # is average mu of each server
## SMALLDRY
SMALLDRY_QUEUE_LENGTH = math.inf
SMALLDRY_SERVER_NUM = 4                 # 1 server only (arbitrary)
SMALLDRY_MEAN_SERVICE_RATE = 20         # is average mu of each server
## IRONFOLD
IRONFOLD_QUEUE_LENGTH = math.inf
IRONFOLD_SERVER_NUM = 4                 # 1 server only (arbitrary)
IRONFOLD_MEAN_SERVICE_RATE = 30         # is average mu of each server
## CHECKOUT
CHECKOUT_QUEUE_LENGTH = math.inf
CHECKOUT_SERVER_NUM = 1                 # 1 server only (arbitrary)
CHECKOUT_MEAN_SERVICE_RATE = 50         # is average mu of each server

# There are 3 types of order:
# 0. Only wash
# 1. Only dry
# 2. Wash & dry
#
# Each order will have a random exponentially distributed weight, config 
# for such distribution is chosen below. We will classify them into weight 
# class:
# 0. [0kg, 10kg)
# 1. [10kg, +inf kg)
#
# Each order will have a expected waiting time, if the real waiting time 
# is larger than this value, the order is considered completed late. For 
# now this time will be x2 the total required time without queue blocking 
# (using worst time for between big and small machine).

class OrderType(Enum):
    WASH = 0
    DRY = 1
    WASHDRY = 2
 

class OrderWeightClass(Enum):
    SMALL = 0       # [0, 10kg)
    BIG = 1         # [10, +inf kg)

# WORKLOAD CONFIG
## ORDER TYPES
ORDER_TYPE_WEIGHT_FACTOR = [
    1,              # for only wash
    2,              # for only dry
    1,              # for wash & dry
]
## ORDER WEIGHTS
ORDER_WEIGHT_EXPONENTIAL_LAMBDA = 0.1    # chosen arbitrarily
## ORDER EXPECTED WAITING TIME
ORDER_EXPECTED_WAITING_TIME = [
    3.0 * etime for etime in
    [
        # for wash, time includes checkin, wash, checkout
        (1/CHECKOUT_MEAN_SERVICE_RATE) 
            + max(1/BIGWASH_MEAN_SERVICE_RATE, 1/SMALLWASH_MEAN_SERVICE_RATE)
            + (1/CHECKOUT_MEAN_SERVICE_RATE),
        # for dry, time includes checkin, dry, iron/fold, checkout
        (1/CHECKOUT_MEAN_SERVICE_RATE) 
            + max(1/BIGDRY_MEAN_SERVICE_RATE, 1/SMALLDRY_MEAN_SERVICE_RATE)
            + (1/IRONFOLD_MEAN_SERVICE_RATE)
            + (1/CHECKOUT_MEAN_SERVICE_RATE),
        # for wash+dry, time includes checkin, wash, dry, iron/fold, checkout
        (1/CHECKOUT_MEAN_SERVICE_RATE) 
            + max(1/BIGWASH_MEAN_SERVICE_RATE, 1/SMALLWASH_MEAN_SERVICE_RATE)
            + max(1/BIGDRY_MEAN_SERVICE_RATE, 1/SMALLDRY_MEAN_SERVICE_RATE)
            + (1/IRONFOLD_MEAN_SERVICE_RATE)
            + (1/CHECKOUT_MEAN_SERVICE_RATE),
    ]
]

class ServiceType(Enum):
    CI = auto()
    WB = auto()
    WS = auto()
    DB = auto()
    DS = auto()
    IF = auto()
    CO = auto()

class EventRecord:
    def __init__(self, type: ServiceType, queued_time = 0.0, start_time = 0.0, stop_time = 0.0) -> None:
        self.type = type 
        self.queued_time = queued_time 
        self.start_time = start_time 
        self.stop_time = stop_time 

    def __str__(self) -> str:
        return f'({self.type}, {self.queued_time}, {self.start_time}, {self.stop_time})'

    def __repr__(self) -> str:
        return self.__str__()

class LaundryOrder:
    # various state for order, representing current status of order
    class State(Enum):
        START = auto()
        CI = auto()
        WB = auto()
        WS = auto()
        WB_DB = auto()
        WS_DX = auto()
        DB = auto()
        DS = auto()
        IF = auto()
        CO = auto()
        END = auto()
    
    def __init__(self, id="order_id") -> None:
        self.type = rnd.choices(list(OrderType), weights=ORDER_TYPE_WEIGHT_FACTOR)[0]
        self.weight = rnd.expovariate(ORDER_WEIGHT_EXPONENTIAL_LAMBDA)
        self.weight_class = OrderWeightClass.SMALL if self.weight < 10 else OrderWeightClass.BIG
        self.expected_service_time = ORDER_EXPECTED_WAITING_TIME[self.type.value]
        self.records = [ ]      # for storing EventRecord history
        self.current_state = self.State.START
        self.id = id
        self.finished = False
        self.finished_late = False

    def __str__(self) -> str:
        return f'{self.id}, {self.weight} Kg, {self.expected_service_time} m, {self.records}'

    def __repr__(self) -> str:
        return self.__str__()

class ServiceProvider:
    def __init__(self, env, type: ServiceType, server_num, service_rate, max_queue_len) -> None:
        self.env = env
        self.type = type
        self.server_num = server_num
        self.server_resource = sp.Resource(env, capacity=self.server_num)
        self.service_rate = service_rate 
        self.max_queue_len = max_queue_len
        # statistics
        self.order_arrival = 0        # number of order try to queue
        self.order_admitted = 0       # number of order successfully queued
        self.order_waiting_time = 0 
        self.order_service_time = 0
        self.order_response_time = 0

    def do_process(self, laundry_order):
        self.order_admitted += 1
        with self.server_resource.request() as res:
            queued_time = env.now
            yield res       # wait for resource to become available
            start_time = env.now
            service_time = rnd.expovariate(self.service_rate)
            yield self.env.timeout(service_time)      # do process
            stop_time = env.now
            laundry_order.records.append(EventRecord(self.type, queued_time, start_time, stop_time))
            self.order_waiting_time += start_time - queued_time
            self.order_service_time += service_time
            self.order_response_time += stop_time - queued_time

    def add_to_queue(self, laundry_order):
        self.order_arrival += 1
        if len(self.server_resource.queue) < self.max_queue_len:
            yield self.env.process(self.do_process(laundry_order))
            return True
        else:
            return False

    def is_queue_full(self):
        return len(self.server_resource.queue) >= self.max_queue_len

class LaundrySystem:
    def __init__(self, env) -> None:
        self.checkin = ServiceProvider(env, ServiceType.CI, CHECKIN_SERVER_NUM,
                                       CHECKIN_MEAN_SERVICE_RATE, CHECKIN_QUEUE_LENGTH)
        self.bigwash = ServiceProvider(env, ServiceType.WB, BIGWASH_SERVER_NUM,
                                       BIGWASH_MEAN_SERVICE_RATE, BIGWASH_QUEUE_LENGTH)
        self.smallwash = ServiceProvider(env, ServiceType.WS, SMALLWASH_SERVER_NUM,
                                         SMALLWASH_MEAN_SERVICE_RATE, SMALLWASH_QUEUE_LENGTH)
        self.bigdry = ServiceProvider(env, ServiceType.DB, BIGDRY_SERVER_NUM,
                                      BIGDRY_MEAN_SERVICE_RATE, BIGDRY_QUEUE_LENGTH)
        self.smalldry = ServiceProvider(env, ServiceType.DS, SMALLDRY_SERVER_NUM,
                                        SMALLDRY_MEAN_SERVICE_RATE, SMALLDRY_QUEUE_LENGTH)
        self.ironfold = ServiceProvider(env, ServiceType.IF, IRONFOLD_SERVER_NUM,
                                        IRONFOLD_MEAN_SERVICE_RATE, IRONFOLD_QUEUE_LENGTH)
        self.checkout = ServiceProvider(env, ServiceType.CO, CHECKOUT_SERVER_NUM,
                                        CHECKOUT_MEAN_SERVICE_RATE, CHECKOUT_QUEUE_LENGTH)
        self.env = env
        self.order = 0
        self.order_finished = 0
        self.order_finished_late = 0
        self.env.process(self.generate_laundry_order())

    def generate_laundry_order(self):
        for id in range(POPULATION):
            self.order += 1
            new_order = LaundryOrder(str(id))
            env.process(self.process_laundry_order(new_order))
            yield self.env.timeout(rnd.expovariate(SOURCE_LAMBDA))       # wait for interarrival time
        
    def process_laundry_order(self, laundry_order):
        while True:
            match laundry_order.current_state:
                case LaundryOrder.State.START:
                    log.debug(f'{laundry_order.id} start')
                    laundry_order.current_state = LaundryOrder.State.CI
                case LaundryOrder.State.CI:
                    log.debug(f'{laundry_order.id} checkin start')
                    yield self.env.process(self.process_laundry_order_checkin(laundry_order))
                    log.debug(f'{laundry_order.id} checkin end')
                case LaundryOrder.State.WB:
                    log.debug(f'{laundry_order.id} big wash start')
                    yield self.env.process(self.process_laundry_order_bigwash(laundry_order))
                    log.debug(f'{laundry_order.id} big wash end')
                case LaundryOrder.State.WS:
                    log.debug(f'{laundry_order.id} small wash start')
                    yield self.env.process(self.process_laundry_order_smallwash(laundry_order))
                    log.debug(f'{laundry_order.id} small wash end')
                case LaundryOrder.State.WB_DB:
                    log.debug(f'{laundry_order.id} big wash big dry start')
                    yield self.env.process(self.process_laundry_order_bigwash_bigdry(laundry_order))
                    log.debug(f'{laundry_order.id} big wash big dry end')
                case LaundryOrder.State.WS_DX:
                    log.debug(f'{laundry_order.id} small wash x dry start')
                    yield self.env.process(self.process_laundry_order_smallwash_xdry(laundry_order))
                    log.debug(f'{laundry_order.id} small wash x dry end')
                case LaundryOrder.State.DB:
                    log.debug(f'{laundry_order.id} big dry start')
                    yield self.env.process(self.process_laundry_order_bigdry(laundry_order))
                    log.debug(f'{laundry_order.id} big dry end')
                case LaundryOrder.State.DS:
                    log.debug(f'{laundry_order.id} small dry start')
                    yield self.env.process(self.process_laundry_order_smalldry(laundry_order))
                    log.debug(f'{laundry_order.id} small dry end')
                case LaundryOrder.State.IF:
                    log.debug(f'{laundry_order.id} iron and fold start')
                    yield self.env.process(self.process_laundry_order_ironfold(laundry_order))
                    log.debug(f'{laundry_order.id} iron and fold end')
                case LaundryOrder.State.CO:
                    log.debug(f'{laundry_order.id} checkout start')
                    yield self.env.process(self.process_laundry_order_checkout(laundry_order))
                    log.debug(f'{laundry_order.id} checkout end')
                case LaundryOrder.State.END:
                    log.debug(f'{laundry_order.id} finish' if laundry_order.finished else
                              f'{laundry_order.id} stop')
                    if laundry_order.finished:
                        log.info(laundry_order)     # on log level higher than DEBUG, only this run
                    self.collect_statistics(laundry_order)
                    break       # break out of further order processing 
                case _:
                    raise Exception('Unknown laundry order state', laundry_order.current_state)

    def process_laundry_order_checkin(self, laundry_order):
        proc = yield self.env.process(self.checkin.add_to_queue(laundry_order))
        if proc is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            match laundry_order.type, laundry_order.weight_class:
                case OrderType.WASH, OrderWeightClass.BIG:
                    laundry_order.current_state = LaundryOrder.State.WB
                case OrderType.WASH, OrderWeightClass.SMALL:
                    if not self.smallwash.is_queue_full():
                        laundry_order.current_state = LaundryOrder.State.WS
                    else:
                        laundry_order.current_state = LaundryOrder.State.WB
                case OrderType.DRY, OrderWeightClass.BIG:
                    laundry_order.current_state = LaundryOrder.State.DB
                case OrderType.DRY, OrderWeightClass.SMALL:
                    if not self.smalldry.is_queue_full():
                        laundry_order.current_state = LaundryOrder.State.DS
                    else:
                        laundry_order.current_state = LaundryOrder.State.DB
                case OrderType.WASHDRY, OrderWeightClass.BIG:
                    laundry_order.current_state = LaundryOrder.State.WB_DB
                case OrderType.WASHDRY, OrderWeightClass.SMALL:
                    if not self.smallwash.is_queue_full():
                        laundry_order.current_state = LaundryOrder.State.WS_DX
                    else:
                        laundry_order.current_state = LaundryOrder.State.WB_DB
                case _:
                    raise Exception('Unknown laundry order type and weight class combination',
                                    laundry_order.type, laundry_order.weight_class)

    def process_laundry_order_bigwash(self, laundry_order):
        assert laundry_order.current_state is LaundryOrder.State.WB
        admitted = yield self.env.process(self.bigwash.add_to_queue(laundry_order))
        if admitted is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            laundry_order.current_state = LaundryOrder.State.CO

    def process_laundry_order_smallwash(self, laundry_order):
        assert laundry_order.current_state is LaundryOrder.State.WS
        admitted = yield self.env.process(self.smallwash.add_to_queue(laundry_order))
        if admitted is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            laundry_order.current_state = LaundryOrder.State.CO

    def process_laundry_order_bigwash_bigdry(self, laundry_order):
        assert laundry_order.current_state is LaundryOrder.State.WB_DB
        admitted = yield self.env.process(self.bigwash.add_to_queue(laundry_order))
        if admitted is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            laundry_order.current_state = LaundryOrder.State.DB

    def process_laundry_order_smallwash_xdry(self, laundry_order):
        assert laundry_order.current_state is LaundryOrder.State.WS_DX
        admitted = yield self.env.process(self.smallwash.add_to_queue(laundry_order))
        if admitted is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            if not self.smalldry.is_queue_full():
                laundry_order.current_state = LaundryOrder.State.DS
            else:
                laundry_order.current_state = LaundryOrder.State.DB

    def process_laundry_order_bigdry(self, laundry_order):
        assert laundry_order.current_state is LaundryOrder.State.DB
        admitted = yield self.env.process(self.bigdry.add_to_queue(laundry_order))
        if admitted is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            laundry_order.current_state = LaundryOrder.State.IF

    def process_laundry_order_smalldry(self, laundry_order):
        assert laundry_order.current_state is LaundryOrder.State.DS
        admitted = yield self.env.process(self.smalldry.add_to_queue(laundry_order))
        if admitted is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            laundry_order.current_state = LaundryOrder.State.IF

    def process_laundry_order_ironfold(self, laundry_order):
        assert laundry_order.current_state is LaundryOrder.State.IF
        admitted = yield self.env.process(self.ironfold.add_to_queue(laundry_order))
        if admitted is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            laundry_order.current_state = LaundryOrder.State.CO

    def process_laundry_order_checkout(self, laundry_order):
        assert laundry_order.current_state is LaundryOrder.State.CO
        admitted = yield self.env.process(self.checkout.add_to_queue(laundry_order))
        if admitted is False:        # adding order to queue failed
            laundry_order.current_state = LaundryOrder.State.END
        else:
            laundry_order.current_state = LaundryOrder.State.END
            laundry_order.finished = True
            self.order_finished += 1

            total_service_time = sum(
                [record.stop_time - record.queued_time for record in laundry_order.records]
            )
            if total_service_time > laundry_order.expected_service_time:
                self.order_finished_late += 1
                laundry_order.finished_late = True

    def collect_statistics(self, laundry_order):
        pass

# init the environment
env = sp.Environment()
sys = LaundrySystem(env)

# print env time periodically for long running simulation
def print_env_time(env):
    while True:
        print('Simulation time:', env.now)
        yield env.timeout(200)
env.process(print_env_time(env))

env.run(until=MAX_SIMULATION_TIME)

# print statistics after running simulation
queue_list = [
    (sys.checkin, 'Checkin'),
    (sys.bigwash, 'Big Wash'),
    (sys.smallwash, 'Small Wash'),
    (sys.bigdry, 'Big Dry'),
    (sys.smalldry, 'Small Dry'),
    (sys.ironfold, 'Iron & Fold'),
    (sys.checkout, 'Checkout'),
]

for queue, queue_name in queue_list:
    print(queue_name)
    print('Arrivals'.rjust(20), ':', queue.order_arrival)
    print('Admitted'.rjust(20), ':', queue.order_admitted)
    print('Utilization'.rjust(20), ':', queue.order_service_time / (queue.server_num * env.now))
    print('Mean waiting time'.rjust(20), ':', 
          0 if queue.order_admitted == 0 else queue.order_waiting_time / queue.order_admitted)
    print('Mean service time'.rjust(20), ':', 
          0 if queue.order_admitted == 0 else queue.order_service_time / queue.order_admitted)
    print('Mean response time'.rjust(20), ':', 
          0 if queue.order_admitted == 0 else queue.order_response_time / queue.order_admitted)
    print('------------------------------------------------------------')

# print stats of whole system
print('System')
print('Jobs finished'.rjust(20), ':', f'{sys.order_finished}/{sys.order}')
print('Jobs finished late'.rjust(20), ':', f'{sys.order_finished_late}/{sys.order}')
print('------------------------------------------------------------')
