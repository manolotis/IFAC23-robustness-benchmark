import numpy as np
import matplotlib.pyplot as plt

path_motionCNN = "//home/manolotis/sandbox/robustness_benchmark/motionCNN/evaluations/xception71.npz"
path_multipathPP = "/home/manolotis/sandbox/robustness_benchmark/multipathPP/evaluations/final_RoP_Cov_Single__0f1746c.npz"
path_multipathPP_2 = "/home/manolotis/sandbox/robustness_benchmark/multipathPP/evaluations/final_RoP_Cov_Single_lr4e-4__65c803f.npz"

eval_motionCNN = dict(np.load(path_motionCNN, allow_pickle=True))
eval_multipathPP = dict(np.load(path_multipathPP, allow_pickle=True))
eval_multipathPP_2 = dict(np.load(path_multipathPP_2, allow_pickle=True))

t = np.arange(0.1, 8.01, 0.1)

# print(dict(eval_motionCNN))
for agent in eval_motionCNN.keys():
    eval_motionCNN[agent] = eval_motionCNN[agent].item()
    eval_multipathPP[agent] = eval_multipathPP[agent].item()
    eval_multipathPP_2[agent] = eval_multipathPP_2[agent].item()
    plt.figure(figsize=(10, 6))
    plt.plot(t, eval_motionCNN[agent]["minADE"], label="MotionCNN")
    plt.plot(t, eval_multipathPP[agent]["minADE"], label="MultiPath++")
    plt.plot(t, eval_multipathPP_2[agent]["minADE"], label="MultiPath++ lr 4e-4")

    plt.title(agent)
    plt.xlabel("Time [s]")
    plt.ylabel(f"minADE [m] ({eval_motionCNN[agent]['count']} agents)")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
