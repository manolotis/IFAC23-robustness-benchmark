import numpy as np
import matplotlib.pyplot as plt

path_cv = "/home/manolotis/sandbox/robustness_benchmark/physicsBased/evaluations/cv.npz"
path_lstm = "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/evaluations/lstm_1x512__0303335.npz"
path_motionCNN = "/home/manolotis/sandbox/robustness_benchmark/motionCNN/evaluations/xception71.npz"
path_multipathPP = "/home/manolotis/sandbox/robustness_benchmark/multipathPP/evaluations/final_RoP_Cov_Single__0f1746c.npz"
path_multipathPP_2 = "/home/manolotis/sandbox/robustness_benchmark/multipathPP/evaluations/final_RoP_Cov_Single_lr4e-4__65c803f.npz"

eval_cv = dict(np.load(path_cv, allow_pickle=True))
eval_lstm = dict(np.load(path_lstm, allow_pickle=True))
eval_motionCNN = dict(np.load(path_motionCNN, allow_pickle=True))
eval_multipathPP = dict(np.load(path_multipathPP, allow_pickle=True))
eval_multipathPP_2 = dict(np.load(path_multipathPP_2, allow_pickle=True))

base_lstm_path = "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/evaluations/lstm_{}x{}__0303335.npz"
lstm_evals = []
lstm_labels = []
for n in range(1,4):
    for h in [64,128,256,512]:
        lstm_evals.append(dict(np.load(base_lstm_path.format(n, h), allow_pickle=True)))
        lstm_labels.append(f"LSTM {n}x{h}")

evaluations_to_plot = [  # (evaluation, label)
    (eval_cv, "CV"),
    (eval_lstm, "LSTM 1x512"),
    (eval_motionCNN, "MotionCNN"),
    (eval_multipathPP, "MultiPath++"),
    (eval_multipathPP_2, "MultiPath++ lr 4e-4"),
]
evaluations_to_plot = list(zip(lstm_evals, lstm_labels)) + [evaluations_to_plot[0]]
evaluations_to_plot = list(zip(lstm_evals, lstm_labels))

t = np.arange(0.1, 8.01, 0.1)

# print(dict(eval_motionCNN))
for agent in eval_motionCNN.keys():
    plt.figure(figsize=(10, 6))

    for eval, label in evaluations_to_plot:
        eval[agent] = eval[agent].item()
        plt.plot(t, eval[agent]["minADE"], label=label)

    plt.title(agent)
    plt.xlabel("Time [s]")
    plt.ylabel(f"minADE [m] ({eval[agent]['count']} agents)")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
