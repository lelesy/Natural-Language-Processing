d = 0.7

PR_A = (1-d)
PR_D = (1-d)


def ranking():
    PR_E = 0
    last_PR_B = 0
    PR_B = 1
    iter = 0
    while(abs(last_PR_B - PR_B)!=0):
        last_PR_B = PR_B
        PR_F = (1-d) + d*(PR_D + PR_E)
        PR_E = (1-d) + d*PR_F
        PR_C = (1-d) + d*(PR_A+PR_E+PR_F)
        PR_B = (1-d) + d*(PR_C)
        print "  ====  "
        print PR_A
        print PR_D
        print PR_F
        print PR_E
        print PR_C
        print PR_B
        print last_PR_B
        print iter
        print "  ====  "
        iter+=1

ranking()
