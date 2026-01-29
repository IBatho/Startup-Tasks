import simpy
import random

RANDOM_SEED = 42
INTER_ARRIVAL_TIME = 3.0
SERVICE_TIME = 4.0
NUM_JOBS = 20

class Metrics:
    def __init__(self):
        self.arrival_times = []
        self.start_service_times = []
        self.departure_times = []

    def record(self, arrival, start_service, departure):
        self.arrival_times.append(arrival)
        self.start_service_times.append(start_service)
        self.departure_times.append(departure)

    def summary(self):
        n = len(self.arrival_times)
        if n == 0:
            return {}
        
        waiting_times = [s - a for a, s in zip(self.arrival_times, self.start_service_times)]
        system_times = [d - a for a, d in zip(self.arrival_times, self.departure_times)]

        return {
            "number_jobs": n,
            "avg_wait_time": sum(waiting_times) / n,
            "max_wait_time": max(waiting_times),
            "avg_system_time": sum(system_times) / n,
            "max_system_time": max(system_times),
        }
    
def job(env, name, machine, metrics):
    arrival_time = env.now
    print(f"{arrival_time:5.1f} - Job {name} arrives")

    with machine.request() as req:
        yield req
        start_service_time = env.now
        print(f"{start_service_time:5.1f} - Job {name} starts service")
        service_time = random.expovariate(1.0 / SERVICE_TIME) # random service time with an average of SERVICE_TIME
        yield env.timeout(service_time)
        departure_time = env.now
        print(f"{departure_time:5.1f} - Job {name} departs")

        metrics.record(arrival_time, start_service_time, departure_time)

def job_generator(env, machine, metrics):
    for i in range(NUM_JOBS):
        name = f"Job_{i+1}"
        env.process(job(env, name, machine, metrics))
        #inter_arrival = INTER_ARRIVAL_TIME
        inter_arrival = random.expovariate(1.0 / INTER_ARRIVAL_TIME)
        yield env.timeout(inter_arrival)

def run_simulation():
    random.seed(RANDOM_SEED)

    env = simpy.Environment()
    machine = simpy.Resource(env, capacity=1)
    metrics = Metrics()
    env.process(job_generator(env, machine, metrics))
    env.run()
    summary = metrics.summary()
    print("\nSimulation Summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    run_simulation()