import time
import random

from models.knapsack_solver import KnapsackSolver01, KnapsackSolverFractional

if __name__ == '__main__':
    # for mnist-cnn, n=40; for mobile-net2, n=788, when split into 5
    w_s = [[3, 4, 5], [3, 4, 5] * 13, [3, 4, 5] * 263]
    val_s = [[30, 50, 60], [round(random.random() * 10) for i in range(39)], [round(random.random() * 10) for i in range(789)]]
    C_s = [10, 12, 8]
    for test_i in range(len(w_s)):
        w = w_s[test_i]
        val = val_s[test_i]
        C = C_s[test_i]
        n = len(w)

        knapsack_solver = KnapsackSolver01(value_sum_max=sum(val), item_num_max=len(w), weight_max=sum(w))

        print(f"len is {len(w)}")
        start = time.clock()
        for i in range(1):
            res = knapsack_solver.found_max_value(w, val, C)
        end = time.clock()
        # print(res)
        print(end - start)

        # start = time.clock()
        # for i in range(500):
        #     res = knapsack_solver.found_max_value_iter(w, val, C)
        # end = time.clock()
        # print(res)
        # print(end - start)
        # knapsack_solver = KnapsackSolverFractional(item_num_max=len(w), weight_max=sum(w))

        # print(knapsack_solver.found_max_value(w, val, C))

        # print("\n")
