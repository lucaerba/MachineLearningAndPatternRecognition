import sys
import numpy as np

class Bus:
    def __init__(self,id,route):
        self.id = id
        self.route = route
        self.lectures = []

    def add_lecture(self,x,y,time):
        self.lectures.append([x,y,time])

    def total_distance(self):
        total = 0
        x_prec, y_prec, t = self.lectures[0]
        for x,y,t in self.lectures[1:]:
            total += np.sqrt((int(x) - int(x_prec))**2 + (int(y) - int(y_prec))**2)
            x_prec, y_prec = x, y
        return total

    def mean_velocity(self):
        velocity = []
        total_time = []
        for i in list(range(1,len(self.lectures))):
            x_prec, y_prec = self.lectures[i-1][0:2]
            x, y = self.lectures[i][0:2]
            total_dist = np.sqrt((int(x) - int(x_prec)) ** 2 + (int(y) - int(y_prec)) ** 2)
            time_start = self.lectures[i-1][2]
            t = self.lectures[i][2]
            total_time.append(int(t) - int(time_start))
            velocity.append(total_dist / total_time[i-1])
        mean_vel = np.dot(velocity,total_time) / sum(total_time)
        return mean_vel

def read_file(file):
    f = open(file,'r')
    for line in f:
        id,route,x,y,time = line.split()

        if id not in buses:
            buses[id] = Bus(id,route)

        buses[id].add_lecture(x, y, time)


input_file = sys.argv[1]
param = sys.argv[2]
buses = {}
read_file(input_file)
if param == '-b':
    bus_id = sys.argv[3]
    print(bus_id + " - Total distance: " + str(buses[bus_id].total_distance()))
elif param == '-l':
    line_id = sys.argv[3]
    total_mean_velocity = 0
    total_busses = 0
    for b in buses.values():
        if(int(b.route) == int(line_id)):
            total_busses += 1
            total_mean_velocity += b.mean_velocity()
    print(line_id + " - Avg Speed: " + str(total_mean_velocity/total_busses))


