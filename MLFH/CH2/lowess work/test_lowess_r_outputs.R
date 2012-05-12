# test_lowess_r_output.R
#
# Generate outputs for unit tests
# for lowess function in cylowess.pyx
#
# May 2012
#
 
# test_simple
x = 0:19
# Standard Normal noise
noise = c(-0.76741118, -0.30754369,  
           0.39950921, -0.46352422, -1.67081778,
           0.6595567 ,  0.66367639, -2.04388585,  
           0.8123281 ,  1.45977518,
           1.21428038,  1.29296866,  0.78028477, 
           -0.2402853 , -0.21721302,
           0.24549405,  0.25987014, -0.90709034, 
           -1.45688216, -0.31780505)

y = x + noise

test.simple.out = lowess(x, y, delta = 0, iter = 3)

# Print comma separated results (to paste into test file)
print("Simple test outputs")
paste(round(test.simple.out$y, 10), collapse = ", ")


# test_iter
x = 0:19
# Cauchy noise
noise = c(1.86299605, -0.10816866,  1.87761229, 
          -3.63442237,  0.30249022,
          1.03560416,  0.21163349,  1.14167809, 
          -0.00368175, -2.08808987,
          0.13065417, -1.8052207 ,  0.60404596, 
          -2.30908204,  1.7081412 ,
          -0.54633243, -0.93107948,  1.79023999,  
          1.05822445, -1.04530564)

test.no.iter.out = lowess(x, y, delta = 0, iter = 0)
test.3.iter.out = lowess(x, y, delta = 0, iter = 3)

print("Iter test outputs")
paste(round(test.no.iter.out$y, 10), collapse = ", ")
paste(round(test.3.iter.out$y, 10), collapse = ", ")


# test_frac
x = seq(-2*pi, 2*pi, length = 30)

# normal noise
noise = c( 1.62379338, -1.11849371,  1.60085673,  
           0.41996348,  0.70896754,
           0.19271408,  0.04972776, -0.22411356,  
           0.18154882, -0.63651971,
           0.64942414, -2.26509826,  0.80018964,  
           0.89826857, -0.09136105,
           0.80482898,  1.54504686, -1.23734643, 
           -1.16572754,  0.28027691,
           -0.85191583,  0.20417445,  0.61034806, 
           0.68297375,  1.45707167,
           0.45157072, -1.13669622, -0.08552254, 
           -0.28368514, -0.17326155)

y = sin(x) + noise

frac.2_3.out = lowess(x, y, f = 2/3, delta = 0, iter = 3)
frac.1_5.out = lowess(x, y, f = 1/5, delta = 0, iter = 3)

print("Frac test outputs")
paste(round(frac.2_3.out$y, 10), collapse=", ")
paste(round(frac.1_5.out$y, 10), collapse=", ")


# test_delta
# Load mcycle motorcycle collision data
library(MASS)
data(mcycle)

delta.0.out = with(mcycle, lowess(times, accel, delta = 0.0))
delta.0_1.out = with(mcycle, lowess(times, accel, delta = 0.1))
delta.default.out = with(mcycle, lowess(times, accel))

print("mcycle x values")
paste(mcycle$times, collapse = ", ")

print("mcycle y values")
paste(mcycle$accel, collapse = ", ")

print("Delta test outputs")
paste(round(delta.0.out$y, 10), collapse = ", ")

paste(round(delta.0_1.out$y, 10), collapse = ", ")
paste(round(delta.default.out$y, 10), collapse = ", ")
