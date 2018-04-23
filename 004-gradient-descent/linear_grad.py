import matplotlib.pyplot as plt
import numpy as np

x = np.array(list(range(0, 11)))
y = np.array(list(map(lambda n: n*n, x)))

def compute_cost_function(m, t0, t1, x, y):
    return 1/2/m * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])
def gradient_descent(x, y, alpha = 0.01, ep=0.0001, max_iter=1500):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta
    t0 = 0
    t1 = 0

    # total error, J(theta)
    J = compute_cost_function(m, t0, t1, x, y)
    # print('J=', J);
    # Iterate Loop
    num_iter = 0
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = compute_cost_function(m, t0, t1, x, y)
        # print ('J = ', e)
        J = e   # update error
        iter += 1  # update iter

        if iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return t0,t1

def graph_points(data_x, data_y):
    for x, y in zip(data_x, data_y):
        plt.scatter(x, y, c='r')
def graph_line(data_x, m, b):
    line = [ (m*x) + b for x in data_x]
    plt.plot(data_x, line, c='b')

def squared_error(y_points, regression_line):
    return sum( (regression_line - y_points)**2)

def coeff_of_determination(y_points, regression_line):
    y_mean_line = [np.mean(y_points) for y in regression_line]
    squared_error_regr = squared_error(y_points, regression_line)
    squared_error_y_mean = squared_error(y_points, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

graph_points(x, y)
b, m = gradient_descent(x, y)
print(m, b)
graph_line(x, m, b)
print(coeff_of_determination(y, m*x+b))
plt.show()
