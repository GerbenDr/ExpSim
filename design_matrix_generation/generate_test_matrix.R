# Import the library
library(AlgDesign)

# --------------------------------------------------------------------------------------------

## Set the factor limits and step sizes
# NOTE: step sizes is only relevant for the number of points that the algorithm samples: high number of points (small stepsize), high computation time

# Factor: Alpha (Angle of Attack)
alpha_L = -4
alpha_U = 7
alpha_stepsize = 0.01

# Factor: J (Advance Ratio)
J_L = 1.8
J_U = 2.2
J_stepsize = 0.01

# Factor: delta_e (Elevator Deflection)
# NOTE: delta_e is a binary factor, so we only need to specify the levels
delta_e_L = 0
delta_e_U = 10

# --------------------------------------------------------------------------------------------

## Specify the number of test points
number_of_trials = 100

# --------------------------------------------------------------------------------------------

## Specify the candidate set (using levels specified above)
candidates <- expand.grid(
    alpha = seq(alpha_L, alpha_U, length = ((alpha_U - alpha_L) / alpha_stepsize + 1)),
    J = seq(J_L, J_U, length = ((J_U - J_L) / J_stepsize + 1)),
    delta_e = c(delta_e_L, delta_e_U)
)

# --------------------------------------------------------------------------------------------

# Check the time it takes: start a timer
start <- Sys.time()

# Use the optFederov function to generate the optimal test matrix
# First, the polynomial is specified
# Second, the data (candidates specified above) is specified
# Third, the criterion is specified (I-criterion)
# Fourth, the number of trials is specified (100)
design <- optFederov(
  ~ poly(alpha, 3) + poly(J, 3) + poly(delta_e, 1) + J:alpha,
  data = candidates,
  criterion = "I",
  nTrials = number_of_trials
)

# Check the time it takes: stop the timer
end <- Sys.time()
end - start

# --------------------------------------------------------------------------------------------

## Report the results
# Print the design
print(design)

# Print the time it took
print(end - start)

# Save the design to a csv file
write.csv(design, "design_matrix_generation/raw_test_matrix.csv", row.names = FALSE)
