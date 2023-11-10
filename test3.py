'''
线性规划模型
max x+y+2z
s.t.
    x+2y+3z<=4
    x+y>=1
    x,y,z=0 or 1
'''
import numpy as np
from gurobipy import *

try:
    #模型
    model=Model('mip')

    #变量
    x=model.addMVar((3), vtype=GRB.CONTINUOUS, name='x')

    #参数
    a = np.array([1, 1, 2])
    c1 = np.array([1, 2, 3])
    c2 = np.array([1, 1, 0])

    #目标函数
    model.setObjective(a @ x, GRB.MAXIMIZE)

    #约束
    model.addConstr(c1 @ x <= 4,name='c1')
    model.addConstr(c2 @ x >= 1,name='c2')

    #求解
    model.setParam('outPutFlag',0)#不输出求解日志
    model.optimize()

    #输出
    print('obj=',model.objVal)
    for v in model.getVars():
        print(v.varName,':',v.x)
        
except GurobiError as e:
    print('Error code '+str(e.errno)+':'+str(e))

except AttributeError:
    print('Encountered an attribute error')
