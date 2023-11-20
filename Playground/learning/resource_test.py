import simpy

def print_stats(res, id):
    print(f'From {id}')
    print(f'{res.count} of {res.capacity} slots are allocated.')
    print(f'  Users: {res.users}')
    print(f'  Queued events: {res.queue}')

def user(res, id):
    print_stats(res, id)
    with res.request() as req:
        yield req
        print_stats(res, id)
    print_stats(res, id)

env = simpy.Environment()
res = simpy.Resource(env, capacity=1)

procs = [env.process(user(res, '1')), env.process(user(res, '2'))]
env.run()
