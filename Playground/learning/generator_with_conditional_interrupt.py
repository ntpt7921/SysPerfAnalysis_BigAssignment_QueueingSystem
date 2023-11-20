import simpy
from random import randint

def speaker(env, id):
    try:
        print(f'speaker {id} starts at time {env.now}')
        
        while True:
            time = randint(25, 35)
            if time != 30:
                break

        yield env.timeout(time)

        print(f'speaker {id} ends at time {env.now}')
    except simpy.Interrupt as itr:
        print(f'speaker {id} interrupted at time {env.now}')
        print(itr.cause)

def moderator(env):
    for id in range(3):
        speaker_proc = env.process(speaker(env, id))
        result = yield speaker_proc | env.timeout(30)

        if speaker_proc not in result:
            speaker_proc.interrupt('No time left!')

env = simpy.Environment()
env.process(moderator(env))

env.run()
