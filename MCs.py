import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 10 ** -2  # probability of an individual being infected
N = 10 ** 5  # number of blood samples
trials = 5000  # number of Monte Carlo trials
batch_sizes = [2, 5, 10, 20, 50, 100, 200, 500, 1000]  # batch sizes to test

# 6,7,8,9,10,11,12,13,14
# 2, 5, 10, 20, 50, 100, 200, 500, 1000
# 70,80,90,100,110,120,130

# Function for the testing process
def simulate_testing(p, N, batch_size):
    individual_results = np.random.binomial(1, p, N)  # generate individual test results
    num_batches = N // batch_size

    if N % batch_size == 0:
        batch_results = np.mean(np.reshape(individual_results, (num_batches, batch_size)), axis=1)

    else:  # this is extra: if N not divisible by batch size we make a last batch with the leftovers
        num_batches = (N - (N % batch_size)) // batch_size
        batch_results = np.mean(np.reshape(individual_results[:-(N % batch_size)], (num_batches, batch_size)), axis=1)
        batch_results = np.append(batch_results, np.mean(individual_results[-(N % batch_size):]))

    retests = np.count_nonzero(batch_results)  # number of positive batches

    return num_batches + batch_size * retests  # total number of tests

# Monte Carlo simulation
print('N={:.0e}, p={:.0e}'.format(N,p))
expected_tests = np.zeros(len(batch_sizes))
std_tests = np.zeros(len(batch_sizes))
for i, batch_size in enumerate(batch_sizes):
    #print("batch size: ", batch_size)
    total_tests = np.zeros(trials)
    for j in range(trials):
        total_tests[j] = simulate_testing(p, N, batch_size)
    #print(total_tests[0:10])
    expected_tests[i] = np.mean(total_tests)
    std_tests[i] = np.std(total_tests)
    print('The expected number of tests for k={} is {:.2f} (std={:.2f})'.format(batch_size, expected_tests[i], std_tests[i]))

# Finding the optimal batch size
optimal_batch_size = batch_sizes[np.argmin(expected_tests)]
print('-----------------------\np={:.0e}, N={:.0e}:\n-----------------------\nFrom the batches considered, the optimal batch '
      'size\nis {}, with {:.2f} expected number of tests and std of {:.2f}'.format(p, N, optimal_batch_size, expected_tests[np.argmin(expected_tests)], std_tests[np.argmin(expected_tests)]))
# Plotting the results
plt.errorbar(batch_sizes, expected_tests, yerr=std_tests, fmt='ro', label='MCs curve')
plt.plot(batch_sizes, expected_tests,'r', label='Expected number of tests')
x=np.arange(batch_sizes[0], batch_sizes[-1])
trueval=[]
for i in x:
    trueval.append(N/i+ N*min(1,i*p))
plt.plot(x,trueval,'b--',label='theorized curve',linewidth=1)
plt.xlabel('Batch size')
plt.ylabel('Expected number of tests')
plt.title('MCs averages. P={}'.format(p))
plt.legend(loc='lower right')
plt.show()