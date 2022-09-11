# a-star

This project is a classic algorithms assignment.  The task is to efficiently solve a "slider puzzle" using the [A<sup>*</sup>](https://en.wikipedia.org/wiki/A*_search_algorithm) graph search algorithm.  The problem statement can be found [here](https://coursera.cs.princeton.edu/algs4/assignments/8puzzle/specification.php).  Despite its ancient origins ([1968](https://ieeexplore.ieee.org/document/4082128)), the A<sup>*</sup> algorithm remains useful to this day for tasks such as pathfinding in video games, parsing in natural language processinng, and informational search in query engines.  Personally, though, I find its [Shakey](https://en.wikipedia.org/wiki/Shakey_the_robot) origins to be the most fascinating -- the A<sup>*</sup> algorithm was an early attempt at constructing an artificial agent able to guide its own actions in the natural world, and thus represents a kind of prehistory of the reinforcement learning based agents that are in development today.

We will give a brief summary of the problem statement.  We are given an n x n grid of tiles, each with an integer from 1 to n<sup>2</sup>-1 printed on it.  There is one empty space.  The tiles start out in scrambled order and our job is to slide one tile at a time through the empty space to bring the grid into sorted order (left to right, as you read).  Here is a pictorial example of a 3 x 3 grid being brought into sorted order through a series of legal moves:

![Alt text](puzzle_example.png)

The A<sup>*</sup> algorithm works by searching along paths in the space of all possible boards according to a heuristic that captures the distance of the present board from the start and the distance of the present board from the goal.  Effectively, we want to investigate those boards first that are both closest to the start and closest to the goal, as that will lead us to a shortest path to the objective.  The advantage of this heuristic approach over "blind" methods like Dijkstra's algorithm (which searches all paths in any order) is simple speed.  Using a heuristic effectively narrows the search space by guiding the algorithm along a set of candidate paths most likely to lead to the goal, thereby reducing the time complexity.

We structure the search using a binary heap implementation of a minimum priority queue.  The nodes in our priority queue are structured as follows:

```python
class Node():
	def __init__(self,board, g, prev, heuristic = 'hamming'):
		self.board = board #representation of the current board
		self.g = g #distance from the initial board to the current board
		self.prev = prev #the previous board visited in the search
		if heuristic == 'hamming':
			self.h = board.hamming() #compute the hamming distance to the goal
		if heuristic == 'manhattan':
			self.h = board.manhattan() #compute the manhattan distance to the goal
		self.f = self.g + self.h #the priority is the sum of the distance to the goal and the distance to the start

```  
Each node stores a representation of its board and uses for its priority the sum of the distance from the current board to the start and the distance from the current board to the goal.

Here is our implementation of the binary heap data structure:

```python
class BinaryHeap():
	def __init__(self):
		self.heap = [0] #the zeroth element is unused

	def exch(self,i,j):
		self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

	def swim(self,k):
		while k>1 and self.heap[k//2].f > self.heap[k].f:
			self.exch(k,k//2)
			k = k//2

	def sink(self,k):
		N = len(self.heap) - 1
		while 2*k <= N:
			j = 2*k
			if j < N and self.heap[j].f > self.heap[j+1].f:
				j+=1
			if self.heap[k].f <= self.heap[j].f:
				break
			self.exch(k,j)
			k = j

	def insert(self, node):
		self.heap.append(node)
		N = len(self.heap) - 1
		self.swim(N)

	def delMin(self):
		N = len(self.heap) - 1
		self.exch(1,N)
		m = self.heap.pop()
		self.sink(1)
		return m
```

Of course, we could have used the module `heapq` to implement our priority queue, but it is a good exercise to code up the data structure from scratch to see how it all works -- and fun too!

Lastly, we need a representation of the board.  Here it is:

```python
class Board():
	#An object representing the puzzle board
	def __init__(self,n,board_in=None):
		tmp = np.array([i for i in range(1,n**2)]+[0])
		self.goal = np.reshape(tmp.copy(),[n,n]) # define the goal board
		
		if board_in is None: # initialize the board randomly if no input board is provided
			np.random.shuffle(tmp)
			self.board = np.reshape(tmp,[n,n])

		else: #use the input board if provided
			self.board = board_in

	def dimension(self):
		#returns the size of the board
		return self.board.shape[0]

	def hamming(self):
		#computes the hamming distance to the goal
		return np.sum(1-(self.board==self.goal))

	def manhattan(self):
		#computes the manhattan distance to the goal
		n = self.dimension()
		nyc = 0
		for i in range(n):
			for j in range(n):
				item = self.board[i,j]
				ii, jj = np.nonzero(self.goal==item)
				nyc += abs(ii[0] - i) + abs(jj[0] - j)
		return nyc

	def isGoal(self):
		#returns whether or not the board is at the goal
		return self.hamming()==0

	def notEquals(self, y):
		#checks whether the current board is equivalent to an input board y
		return np.sum(1-(self.board==y))

	def neighbors(self):
		#returns all boards that are neighbors to the current board
		#a neighboring board is one that can be reached from the current board in exactly one move
		flanders = []
		n = self.dimension()
		i,j = np.nonzero(self.board==0)
		i = i[0]
		j = j[0]
		if i + 1 < n:
			tmp = self.board.copy()
			tmp[i,j] = tmp[i+1,j]
			tmp[i+1,j] = 0
			flanders.append(tmp)
		if j + 1 < n:
			tmp = self.board.copy()
			tmp[i,j] = tmp[i,j+1]
			tmp[i,j+1] = 0
			flanders.append(tmp)
		if i - 1 >= 0:
			tmp = self.board.copy()
			tmp[i,j] = tmp[i-1,j]
			tmp[i-1,j] = 0
			flanders.append(tmp)
		if j - 1 >=0:
			tmp = self.board.copy()
			tmp[i,j] = tmp[i,j-1]
			tmp[i,j-1] = 0
			flanders.append(tmp)

		return flanders

	def twin(self):
		#returns a board with one pair of tiles swapped from the original board
		#all boards fall into two classes: (1) the board is solvable, (2) the board is solvable
		#only if we swap a pair of tiles.  This method helps us to find unsolvable boards.
		i,j = 0,0
		ii,jj = 0,1
		tmp = self.board.copy()
		val = tmp[i,j]
		tmp[i,j] = tmp[ii,jj]
		tmp[ii,jj] = val
		return tmp
```

We are now ready to implement the A<sup>*</sup> algorithm.  First we initialize our board and its twin, placing each on a priority queue.

```python
n=3 #define the size of the board (integer)
heur = 'manhattan' #choose either 'manhattan' or 'hamming' for the heuristic
board_init = Board(n)
board_init_twin = Board(n,board_in=board_init.twin())
start = Node(board_init,0,None,heuristic=heur)
start_twin = Node(board_init_twin,0,None,heuristic=heur)
heap = BinaryHeap()
heap_twin = BinaryHeap()

heap.insert(start)
heap_twin.insert(start_twin)
```

The algorithm works by grabbing the lowest priority node from the queue, checking if it is the goal node, and if not adding its neighbors to the queue.  This is repeated until the goal node is reached.

```python
while True:
	#grab the lowest priority nodes
	node = heap.delMin()
	node_twin = heap_twin.delMin()

	#break out of the while statement as soon as either board is solved
	if node.board.isGoal():
		print('Original')
		break
	if node_twin.board.isGoal():
		print('Twin')
		break

	#add neighbors to the queue, being careful to avoid adding the board from which we came
	for x in node.board.neighbors():
		b = Board(n,board_in=x)
		if node.prev is None:
			nn = Node(b,1+node.g,node,heuristic=heur)
			heap.insert(nn)
		elif node.prev.board.notEquals(x):
			nn = Node(b,1+node.g,node,heuristic=heur)
			heap.insert(nn)

	for x in node_twin.board.neighbors():
		b = Board(n,board_in=x)
		if node_twin.prev is None:
			nn = Node(b,1+node_twin.g,node_twin,heuristic=heur)
			heap_twin.insert(nn)
		elif node_twin.prev.board.notEquals(x):
			nn = Node(b,1+node_twin.g,node_twin,heuristic=heur)
			heap_twin.insert(nn)
```
