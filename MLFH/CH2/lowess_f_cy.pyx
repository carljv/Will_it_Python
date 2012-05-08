from libc.math import imax, fmax, fabs
import numpy as np
cimport numpy as np

def lowess(np.ndarray[DTYPE_t, ndim = 1] x,
           np.ndarray[DTYPE_t, ndim = 1] y,
           double frac,
           Py_ssize_t robust_iters,
           double delta)
    cdef:
        Py_ssize_t n = x.size
        # The number of nearest-neighbors in each regression
        # Should be at least 2 (otherwise what to regress?)
        Py_ssize_t k = max(int(frac * n), 2)
        # The indices of the left-most and right-most nearest
        # neighbors. Start at [0,k], will always be k apart.
        Py_ssize_t left_end = 0, 
        Py_ssize_t right_end = k
        # i indexes along the vectors x[i], y[i].
        # j is the index of neighbors for a given i (j = 0..k-1)
        # iter is a counter for robustness iterations.
        Py_ssize i, j, iter

        double range,
               left_width,
               right_width,
               width,
               width999,
               width001,
               dist_i_j,
               weight,
               resid_weight,
               sum_weights
        
        # TODO: Assert k < n (e.g. f < 1)

    for iter in xrange(robust_iters):
        # Find the indices of the leftmost and rightmost neighbors
        # of x_i.  To do so, we shift the neighborhood rightward until
        # x_i is just before the mid-point.
        # (DO RIGHT -1 OR JUST RIGHT??)
        while True:
            while (right_end < n and x[i] > (x[right_end - 1] - x[left_end]) / 2):
                left_end += 1
                right_end += 1

            ### CALL LOWEST  FUNTION -
            ###### 1. Compute and sum weights
            ###### 2. Check if regression to be run
            ###### 3. 
            # Fitted value at x[i]
            if ok_flag == False:
                y_smooth[i] = y[i]
            # If all weights zero, copy over value
            if (last_x_index < i - 1):
                denom = x[i] - x[last_x_index]
                for j in xrange(last_x_index + 1, i):
                    alpha = (x[j] - x[last_x_index]) / denom
                    y_smooth[j] = (alpha * y_smooth[i] +
                                   (1.0 - alpha) * y_smooth[last_x_index])
            last_x_index = i
            cut = x[last_x_index] + delta
            for i in xrange(last_x_index + 1, n):
                if x[i] > cut:
                    break
                if x[i] == x[last_x_index]:
                    y_smooth[i] = y_smooth[last_x_index]
                    last_x_index = i
            i = imax(last_x_index + 1, i - 1)
            if last_x_index >= n:
                break

    # COMPUTE RESIDUAL WEIGHTS
    for i in xrange(n):
        resid[i] = y[i] - y_smooth[i]
    if (iter > robust_iters):
        break
    for i in xrange(n):
        resid_weight[i] = abs(resid[i])
    median_abs_resid = np.median(resid_weight[i])
    for i in xrange(n):
        r = resid_weight[i]
        if (r <= .001 * median_abs_resid):
            resid_weight[i] = 1.
        elif (r > .999 * median_abs_resid):
            resid_weight[i] = 0.
        else:
            resid_weight[i] = (1.0 - (resid_weight / median_abs_resid)**2)**2

           
def lowest():
            # BEGIN `LOWEST` FUNCTION #

    ##### GET WEIGHTS ######
    # Weights are a function of the the distance between points x_i
    # and x_j, D(i, j), and the radius of the neighborhood around x_i
    # rad(i). If D(i, j) / rad(i) ~= 0 (< 0.01%) then weight = 1
    # if D(i, j) / rad(i) ~= 1 (> 99.9%) then weight = 0.
    # Once we find the first point on the right with a D(i,j)
    # > 99.9%*rad(i) then all other points will be, so we no
    # longer have to keep computing -- all those weights will be
    # 0.
    left_width  = x[0] - x[left_end]
    right_width  = x[right_end] - x[0]
    width = = fmax(left_width, right_width)
    width999 = 0.999 * width  
    width001 = 0.001 * width
    sum_weights = 0.0
    for j in xrange(left_end, n):
        dist_i_j = fabs(x[j] - x)
        if dist_i_j <= h999:
            if dist > h001:
                weight[j] = (1.0 - (dist_i_j / width) ** 3) ** 3
            else:
                weight[j] = 1.0
            if use_resid_weights == True
                weight[j] *= resid_weight[j]
                sum_weights += weight[j]
        elif x[j] > x[i]:
            break
    
    alt_right_end = j -1 # `effective` right point if stopped early
    if sum_weights <= 0:
        ok_flag == False
    else:
        ok_flag == True
        for j in xrange(left_end, alt_right_end):
            w[j] /= sum_weights
        if width > 0.0:
            sum_weights = 0.0
            for j in xrange(left_end, alt_right_end):
                sum_weights += weight[j] * x[j]
            b = x[i] - sum_weights
            c = 0.0
            for j in xrange(left_end, alt_right_end):
                c = c + weight[j] * (x[j] - sum_weights) ** 2
                if sqrt(c) > .001 * range
                    b /= c
                    for j in xrange(left_end, alt_right_end):
                        weight[j] *= (1.0 + b * (x[j] - sum_weights))
        y_smooth[i] = 0.0
        for j in xrange(left_end, alt_right_end):
            y_smooth[i] += weight[j]*y[j]

############### END LOWEST FUNCTION #



# OUTLINE
# FOR ITER = 1 to NITERS -
# 1. Set i to 1, last_index = 0; set left_end and right_end to 0, k
# 2. Do until last estimated point index (last_index) > n:
#    1. Update the knn-neighborhood around x_i
#    2. Call LOWEST (LOWESS-ESTIMATE) function
#       Args| x, y, n, x[i], ys[i] (output), left_end, right_end, y_weight (output)
#           | use_res_weight = iter > 1(0), res_weights, ok_flag (output)
#       1. Compute weights
#           1. for j in left_end to right_end, w_j = triweight(dist_i_j / rad_i)
#              except for dist_i_j/rad_i ~= 0 or ~= 1.
#           2. for j > last_weight_index (= argmin(k: w_k = 0 and x_k > x_i) - 1)
#              set w_j = 0.
#           3. if sum of weights = 0, the ok_flag is FALSE
#           4. otherwise
#              1. normalize weights so w_j -> w_j / sum_j(w_j)
#              2. compute wtd_x = sum_j(x_j * w_j)
#              3. compute projection to find ysmooth_i e.g. P_ij
#              4. ysmooth_i = sum_j(P_ij * y_j)
#    3. If ok_flag = FALSE (zero weights) y_smooth_i = y_i
#    4. Otherwise:
#       1. If last_index < i - 1: ysmooth_[last_index to i-1]
#             = lin_interpolate(y_smooth_last_index, y_smooth_i)
#    5. set last_index = i
#    6. if x_i == x_last_index, then ysmooth_i = ysmooth_last_index; set last_index = i
#    7. set i to max(last_index + 1, i - 1)
#    8. Get residual weights: if rw_j = bisquare(|e_j|/6*med(|e|)) adjust for ~1, ~0. 
# 
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
