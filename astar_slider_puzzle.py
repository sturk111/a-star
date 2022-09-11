#solves the slider puzzle problem on an nxn grid using A* search.
#this code implements a min priority queue using a binary heap data structure
import numpy as np

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




#------------------
#START A* ALGORITHM
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







