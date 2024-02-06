import numpy as np
import matplotlib.pyplot as plt

def golden(a,b,error,maxiter,numiter=0,gr=0.618):
  if (b-a<error or numiter>=maxiter):
    return(a,b,numiter)
  d=gr*(b-a)
  x1=a+d
  x2=b-d
  funx1=x1**2
  funx2=x2**2
  if(funx1>funx2):
    return(golden(a,x1,error,maxiter,numiter+1))
  else:
    return(golden(x2,b,error,maxiter,numiter+1))

def ibs(a,b,error,maxiter,numiter=0):
  if (b-a<error or numiter>=maxiter):
    return(a,b,numiter)
  c=(b+a)/2
  funx1=a**2
  funx2=b**2
  if(funx1>funx2):
    return(ibs(c,b,error,maxiter,numiter+1))
  else:
    return(ibs(a,c,error,maxiter,numiter+1))

def run_trials(algorithm, tol_range, num_trials=50):
    avg_iterations = []
    for tol in tol_range:
        total_iterations = 0
        for _ in range(num_trials):
            _, _, iterations = algorithm(-10, 10, tol, 1000)
            total_iterations += iterations
        avg_iterations.append(total_iterations / num_trials)
    return avg_iterations

#graph help from chat gpt
tol_range = np.logspace(-8, 0, 50)
num_trials = 50
ibs_iterations = run_trials(ibs, tol_range, num_trials)
golden_iterations = run_trials(golden, tol_range, num_trials)
plt.plot(np.log10(tol_range), np.log10(ibs_iterations), 'bo-', label='IBS')
plt.plot(np.log10(tol_range), np.log10(golden_iterations), 'ro-', label='GSS')
plt.xlabel('$log_{10}$ tol')
plt.ylabel('$log_{10}$ Avg. Iterations')
plt.legend()
plt.show()
#IBS seems to perform better on average as it takes fewer iterations constanly then the golden section search method
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
def objective_function(v):
    return (v[0] - 1)**2 + (v[1] - 1)**2
v1_range = np.arange(-3, 12.1, 0.1)
v2_range = np.arange(-6, 60.1, 1)
nma_distances = np.zeros((len(v1_range), len(v2_range)))
cg_distances = np.zeros((len(v1_range), len(v2_range)))

for i, v1 in enumerate(v1_range):
    for j, v2 in enumerate(v2_range):
        result_nma = minimize(objective_function, [v1, v2], method='Nelder-Mead', options={'maxiter': 5})
        nma_distances[i, j] = np.linalg.norm(result_nma.x - np.array([1, 1]))
        result_cg = minimize(objective_function, [v1, v2], method='CG', options={'maxiter': 5})
        cg_distances[i, j] = np.linalg.norm(result_cg.x - np.array([1, 1]))

#graph help from chatgpt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(v2_range, v1_range, np.log10(nma_distances), levels=20, cmap='viridis')
plt.title('Nelder-Mead Algorithm')
plt.xlabel('$v_2$')
plt.ylabel('$v_1$')
plt.colorbar(label='$log_{10}$(Squared Distance)')
plt.subplot(1, 2, 2)
plt.contourf(v2_range, v1_range, np.log10(cg_distances), levels=20, cmap='viridis')
plt.title('Conjugate Gradient Algorithm')
plt.xlabel('$v_2$')
plt.ylabel('$v_1$')
plt.colorbar(label='$log_{10}$(Squared Distance)')

plt.tight_layout()
plt.show()