# Import the library
library(AlgDesign)

# --------------------------------------------------------------------------------------------

## Set the factor limits and step sizes
# NOTE: step sizes is only relevant for the number of points that the algorithm samples: high number of points (small stepsize), high computation time

# Factor: Alpha (Angle of Attack)
alpha_L = -4
alpha_U = 7
alpha_stepsize = 0.5

# Factor: J (Advance Ratio)
J_L = 1.6
J_U = 2.4
J_stepsize = 0.1

# Factor: delta_e (Elevator Deflection)
# NOTE: delta_e is a binary factor, so we only need to specify the levels
delta_e_L = -10
delta_e_U = 10

# --------------------------------------------------------------------------------------------

## Specify the number of test points
number_of_trials = 2.3 * 20

# --------------------------------------------------------------------------------------------

## Specify the candidate set (using levels specified above)
candidates <- expand.grid(
    alpha = seq(alpha_L, alpha_U, length = ((alpha_U - alpha_L) / alpha_stepsize + 1)),
    J = seq(J_L, J_U, length = ((J_U - J_L) / J_stepsize + 1)),
    delta_e = c(delta_e_L, delta_e_U)
)

# --------------------------------------------------------------------------------------------
set.seed(123)  # For reproducibility
# block_labels <- rep(paste0("Block ", 1:(number_of_trials / 10)), each = 10)
# candidates <- candidates[1:length(block_labels), ]  # Limit to the required number of trials
# candidates$block <- block_labels

# Check the time it takes: start a timer
start <- Sys.time()

# Use the optFederov function to generate the optimal test matrix
# First, the polynomial is specified
# Second, the data (candidates specified above) is specified
# Third, the criterion is specified (I-criterion)
# Fourth, the number of trials is specified (x number of test points)

# Generate the optimal test matrix
design <- optFederov(
  ~ 1 + I(alpha) + I(J) + I(delta_e) + I(alpha^2) + I(J^2) + I(alpha^3) + I(J^3) + I(alpha*J) + I(alpha*delta_e) + I(J*delta_e) + I(alpha^2*J) + I(alpha^2*delta_e) + I(J^2*alpha) + I(J^2*delta_e) + I(alpha*J*delta_e) + I(alpha^2*J^2) + I(alpha^2*J*delta_e) + I(alpha*J^2*delta_e) + I(alpha^2*J^2*delta_e),
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
