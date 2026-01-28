import simpy
import random

RANDOM_SEED = 42
INTER_ARRIVAL_TIME = 5.0
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