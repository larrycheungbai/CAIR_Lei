import heapq
class FixedSizePriorityQueue:
    def __init__(self, size):
        self.size = size
        self.queue = []
    def push(self, item, priority):
        if len(self.queue) < self.size:
            heapq.heappush(self.queue, (priority, item))
            #print(type(self.queue))
        else:
            min_priority, _ = self.queue[0]
            if priority > min_priority:
                heapq.heappop(self.queue)
                heapq.heappush(self.queue, (priority, item))
    def pop(self):
        if not self.is_empty():
            priority, item = heapq.heappop(self.queue)
            return item, priority
        else:
            raise IndexError("Priority queue is empty")
    def is_empty(self):
        return len(self.queue) == 0



class FixedSizePriorityQueueLZ:
    def __init__(self, size):
        self.size = size
        self.queue = []
    def push(self, priority, item):
        if len(self.queue) < self.size:
            heapq.heappush(self.queue, (priority, item))
            #print(type(self.queue))
        else:
            min_priority, _ = self.queue[0]
            if priority > min_priority:
                heapq.heappop(self.queue)
                heapq.heappush(self.queue, (priority, item))
    def pop(self):
        if not self.is_empty():
            priority, item = heapq.heappop(self.queue)
            return priority,item
        else:
            raise IndexError("Priority queue is empty")
    def is_empty(self):
        return len(self.queue) == 0


pq_lz = FixedSizePriorityQueueLZ(3)
pq_lz.push()


while not pq_lz.empty():
	priority, item = pq_lz.pop()
	print(priority)

