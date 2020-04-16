import queue
import math


class Polygon:
    def __init__(self, lower_list, upper_list, divide_list=[2, 2, 2]):
        numberOfCords = min(len(lower_list), len(upper_list))
        self.Peaks = [[] for x in range(numberOfCords)]
        self.Edges = []
        max_size = []
        cord_sign = []
        for i in range(numberOfCords):
            max_size.append(math.fabs(lower_list[i] - upper_list[i]))
            cord_sign.append(1 if lower_list[i] < upper_list[i] else -1)

        new_divide = []
        if len(divide_list) != numberOfCords:
            number_of_divide = min(len(divide_list), numberOfCords)
            for i in range(number_of_divide):
                new_divide.append(divide_list[i])
                if new_divide[i] < 1:
                    new_divide[i] = 1
            for i in range(numberOfCords - number_of_divide):
                new_divide.append(1)
        else:
            new_divide = divide_list

        offset_list = []
        for i in range(numberOfCords):
            offset_list.append(cord_sign[i] * (max_size[i] / new_divide[i]))

        if numberOfCords == 3:
            for z in range(new_divide[2] + 1):
                for y in range(new_divide[1] + 1):
                    for x in range(new_divide[0] + 1):
                        pointNumber = z * ((new_divide[1] + 1) * (new_divide[0] + 1)) + y * (new_divide[0] + 1) + x
                        self.Peaks[0].append(lower_list[0] + x * offset_list[0])
                        self.Peaks[1].append(lower_list[1] + y * offset_list[1])
                        self.Peaks[2].append(lower_list[2] + z * offset_list[2])
                        if x != new_divide[0]:
                            self.Edges.append([pointNumber, pointNumber + 1])
                        if y != new_divide[1]:
                            self.Edges.append([pointNumber, pointNumber + (new_divide[0] + 1)])
                        if z != new_divide[2]:
                            self.Edges.append([pointNumber, pointNumber + (new_divide[1] + 1) * (new_divide[0] + 1)])
        else:
            for y in range(new_divide[1] + 1):
                for x in range(new_divide[0] + 1):
                    pointNumber = y * (new_divide[0] + 1) + x
                    self.Peaks[0].append(lower_list[0] + x * offset_list[0])
                    self.Peaks[1].append(lower_list[1] + y * offset_list[1])
                    if x != new_divide[0]:
                        self.Edges.append([pointNumber, pointNumber + 1])
                    if y != new_divide[1]:
                        self.Edges.append([pointNumber, pointNumber + (new_divide[0] + 1)])


class QueueSet:
    def __init__(self):
        self.queue = queue.Queue()
        self.set = set()

    def add(self, item):
        if not item in self.set:
            self.set.add(item)
            self.queue.put(item)

    def get(self):
        item = self.queue.get()
        self.set.remove(item)
        return item

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.set):
            return self.get()
        else:
            raise StopIteration

    def clear(self):
        self.set.clear()
        with self.queue.mutex:
            self.queue.queue.clear()

    def copy(self):
        tmp = QueueSet()
        tmp.set = self.set.copy()
        for i in tmp.set:
            tmp.queue.put(i)
        return tmp

    def is_empty(self):
        return False if len(self.set) else True

    def __len__(self):
        return len(self.set)
