import numpy as np

one_dimentional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimentional_array)

two_dimentional_array = np.array([[6,4],[8,7],[3,9]])
print(two_dimentional_array)

sequence_of_integers = np.arange(10, 20)
print(sequence_of_integers)

random_integer = np.random.randint(low=100,high=1000,size=6)
print(random_integer)

random_float = np.random.random(size=6)
print(random_float)

random_float_add = random_float + 10
print(random_float_add)

random_integer_multiply = random_integer * 25
print(random_integer_multiply)

features = np.arange(6, 21)
print(features)
label = 3*features + 4
print(label)

noise = (np.random.random([15]) * 4) - 2
print(noise)
label = ((3*features) + 4) + noise
print(label)

