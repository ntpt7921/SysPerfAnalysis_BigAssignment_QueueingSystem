import simpy

class School:
    def __init__(self, env):
        self.env = env
        self.class_end = env.event()
        self.pupil_procs = [env.process(self.pupil(i)) for i in range(3)]
        self.bell_proc = env.process(self.bell())

    def bell(self):
        for _ in range(2):
            yield self.env.timeout(45)
            self.class_end.succeed()
            self.class_end = self.env.event()
            print() # newline

    def pupil(self, i):
        for _ in range(2):
            print(r'\o/', i, end='')
            yield self.class_end

env = simpy.Environment()
school = School(env)
env.run()
