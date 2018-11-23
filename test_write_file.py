import numpy as np
path = '/home/fs168/NewRetail/src_dir/writelines'

f = open(path, 'w')
f.write('\tminx')
list1 = np.array([1, 2, 3, 4])

a = np.count_nonzero(np.where(list1 > 1))
print(a)
f.close()