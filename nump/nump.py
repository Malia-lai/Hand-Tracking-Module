#np is faster than python with arrays
import numpy as np

# -------operations easier------
# array = np.array([1, 2, 3, 4])
# array = array - 1

# print(array)

# -----------dimentions-----------

array = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]) #row, column
#number of brackets = number of dimensions
array = np.array([[[1, 2], [3, 4], ['a', 'b']],
                 [[5, 6], [7, 8], ['c', 'd']],
                 [[9, 10], [11, 12], ['e', 'f']]])

print(f'Array dimensions: {array.ndim}')
print(f'Array shape: {array.shape}') #(rows, columns, depth)
#indexing 
print(f'print element on array[0, 0, 1]: {array[0, 0, 1]}') #first element of the array

# -----------dimentions-----------

# -----------slicing-----------

#array = np.array(start:End:Step , start:End:Step , start:End:Step) with end exclusive
# array = np.array(::2) count two and select
print(f'print rows reversed: {array[::-1]}')
print(f'print from any row and column but depth is first:\n {array[:,:, 0]}')
print(f'Whenever there is a comma, it skips the row or column:\n {array[:,0:1, 0]}') 
print(f'print from any row and column but skip the first depth:\n {array[:,:, 1:]}') 
print(f'print from any row and step 2 for the columns:\n {array[:,::2]}') 
print(f'print from any row step is 2:\n {array[::2]}')


# -----------slicing-----------
