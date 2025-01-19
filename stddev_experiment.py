# import numpy
import numpy as np

def experiment():
    # Loop over the number of samples
    sample_sites = 8
    n_sample = 3

    # Take a sample of n measurements: take 100 random samples from a normal distribution with mean 0 and standard deviation 1
    n = sample_sites * n_sample
    data = np.random.normal(0, 1, n)

    # Calculate the sum of the squares of the differences between the measurements and the mean
    sum_square = np.sum((data - np.mean(data))**2)
    
    # Divide the sum of the squares by n-1
    std_dev_single_site = np.sqrt(sum_square/(n-1))

    # Print the standard deviation
    print(f"Standard deviation of all samples at single site: {std_dev_single_site}")

    # Create an empty list to store the standard deviations
    sum_squares = []

    for i in range(sample_sites):
        
        # Take a sample of n_sample measurements
        data = np.random.normal(0, 1, n_sample)
        
        # Calculate the sum of the squares of the differences between the measurements and the mean
        sum_square = np.sum((data - np.mean(data))**2)
        
        # Append the sum of the squares to the list
        sum_squares.append(sum_square)
        
    # Calculate the sum of the sum of the squares
    sum_sum_squares = np.sum(sum_squares)

    # Divide the sum of the sum of the squares by 10*n-1
    std_dev_diff_sites = np.sqrt(sum_sum_squares/(sample_sites * n_sample - sample_sites))

    # Print the standard deviation
    print(f"Standard deviation based on different sites: {std_dev_diff_sites}")

    return std_dev_single_site, std_dev_diff_sites

if __name__ == "__main__":
    # Run the method 100 times
    std_dev_single_site = []
    std_dev_diff_sites = []
    
    for i in range(100000):
        std_dev_single_site_, std_dev_diff_sites_ = experiment()
        std_dev_single_site.append(std_dev_single_site_)
        std_dev_diff_sites.append(std_dev_diff_sites_)
        
    # Calculate the standard deviation of the standard deviations
    stddev_std_dev_single_site = np.std(std_dev_single_site)
    stddev_std_dev_diff_sites = np.std(std_dev_diff_sites)
        
    # Calculate the mean of the standard deviations
    std_dev_single_site = np.mean(std_dev_single_site)
    std_dev_diff_sites = np.mean(std_dev_diff_sites)
    
    # Print the standard deviation of the standard deviations
    print(f"Standard deviation of standard deviation of all samples at single site: {stddev_std_dev_single_site:.5f}")
    print(f"Standard deviation of standard deviation based on different sites: {stddev_std_dev_diff_sites:.5f}")
    
    # Calculate the difference between the two means
    diff = std_dev_single_site - std_dev_diff_sites
    
    # Print the mean standard deviations
    print(f"Mean standard deviation of all samples at single site: {std_dev_single_site}")
    print(f"Mean standard deviation based on different sites: {std_dev_diff_sites}")
      
    # Plot the results
    import matplotlib.pyplot as plt
    plt.plot(std_dev_single_site, label='Single Site')
    plt.plot(std_dev_diff_sites, label='Different Sites')
    plt.xlabel('Experiment Number')
    plt.ylabel('Standard Deviation')
    plt.legend()
    
    plt.show()
    
    