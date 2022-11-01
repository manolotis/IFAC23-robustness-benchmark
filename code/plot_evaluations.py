import numpy as np
import matplotlib.pyplot as plt

PROJECT_PATH = "/home/manolotis/sandbox/robustness_benchmark/"

path_cv = f"{PROJECT_PATH}physicsBased/evaluations/cv.npz"
path_lstm_1x128 = f"{PROJECT_PATH}lstmAutoencoder/evaluations/lstm_1x128__0303335.npz"
path_lstm_1x128_no_past = f"{PROJECT_PATH}lstmAutoencoder/evaluations/lstm_1x128__0303335_no_past.npz"
path_lstm_3x128 = f"{PROJECT_PATH}lstmAutoencoder/evaluations/lstm_3x128__0303335.npz"
path_lstm_3x128_no_past = f"{PROJECT_PATH}lstmAutoencoder/evaluations/lstm_3x128__0303335_no_past.npz"

path_motionCNN = f"{PROJECT_PATH}motionCNN/evaluations/xception71.npz"
path_motionCNN_no_road = f"{PROJECT_PATH}motionCNN/evaluations/xception71_no_road.npz"
path_motionCNN_no_past = f"{PROJECT_PATH}motionCNN/evaluations/xception71_no_past.npz"
path_motionCNN_noisy_heading = f"{PROJECT_PATH}motionCNN/evaluations/xception71_noisy_heading.npz"

path_motionCNN_no_road_retrained = f"{PROJECT_PATH}motionCNN/evaluations/xception71_no_road_retrained.npz"
path_motionCNN_no_past_retrained = f"{PROJECT_PATH}motionCNN/evaluations/xception71_no_past_retrained.npz"
path_motionCNN_noisy_heading_retrained = f"{PROJECT_PATH}motionCNN/evaluations/xception71_noisy_heading_retrained.npz"
path_motionCNN_no_road_retrained_unperturbed = f"{PROJECT_PATH}motionCNN/evaluations/xception71_no_road_retrained_unperturbed.npz"
path_motionCNN_no_past_retrained_unperturbed = f"{PROJECT_PATH}motionCNN/evaluations/xception71_no_past_retrained_unperturbed.npz"
path_motionCNN_noisy_heading_retrained_unperturbed = f"{PROJECT_PATH}motionCNN/evaluations/xception71_noisy_heading_retrained_unperturbed.npz"

path_multipathPP = f"{PROJECT_PATH}multipathPP/evaluations/final_RoP_Cov_Single_lr4e-4__65c803f.npz"
path_multipathPP_no_road = f"{PROJECT_PATH}multipathPP/evaluations/final_RoP_Cov_Single_lr4e-4__65c803f_no_road.npz"
path_multipathPP_no_past = f"{PROJECT_PATH}multipathPP/evaluations/final_RoP_Cov_Single_lr4e-4__65c803f_no_past.npz"
path_multipathPP_noisy_heading = f"{PROJECT_PATH}multipathPP/evaluations/final_RoP_Cov_Single_lr4e-4__65c803f_noisy_heading.npz"

eval_cv = dict(np.load(path_cv, allow_pickle=True))
eval_lstm_1x128 = dict(np.load(path_lstm_1x128, allow_pickle=True))
eval_lstm_1x128_no_past = dict(np.load(path_lstm_1x128_no_past, allow_pickle=True))
eval_lstm_3x128 = dict(np.load(path_lstm_3x128, allow_pickle=True))
eval_lstm_3x128_no_past = dict(np.load(path_lstm_3x128_no_past, allow_pickle=True))
eval_motionCNN = dict(np.load(path_motionCNN, allow_pickle=True))
eval_motionCNN_no_road = dict(np.load(path_motionCNN_no_road, allow_pickle=True))
eval_motionCNN_no_past = dict(np.load(path_motionCNN_no_past, allow_pickle=True))
eval_motionCNN_noisy_heading = dict(np.load(path_motionCNN_noisy_heading, allow_pickle=True))

eval_motionCNN_no_road_retrained = dict(np.load(path_motionCNN_no_road_retrained, allow_pickle=True))
eval_motionCNN_no_past_retrained = dict(np.load(path_motionCNN_no_past_retrained, allow_pickle=True))
eval_motionCNN_noisy_heading_retrained = dict(np.load(path_motionCNN_noisy_heading_retrained, allow_pickle=True))
eval_motionCNN_no_road_retrained_unperturbed = dict(np.load(path_motionCNN_no_road_retrained_unperturbed, allow_pickle=True))
eval_motionCNN_no_past_retrained_unperturbed = dict(np.load(path_motionCNN_no_past_retrained_unperturbed, allow_pickle=True))
eval_motionCNN_noisy_heading_retrained_unperturbed = dict(np.load(path_motionCNN_noisy_heading_retrained_unperturbed, allow_pickle=True))


eval_multipathPP = dict(np.load(path_multipathPP, allow_pickle=True))
eval_multipathPP_no_road = dict(np.load(path_multipathPP_no_road, allow_pickle=True))
eval_multipathPP_no_past = dict(np.load(path_multipathPP_no_past, allow_pickle=True))
eval_multipathPP_noisy_heading = dict(np.load(path_multipathPP_noisy_heading, allow_pickle=True))

base_lstm_path = "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/evaluations/lstm_{}x{}__0303335.npz"
lstm_evals = []
lstm_labels = []
# for n in range(1, 4):
#     for h in [64, 128, 256, 512]:
#         lstm_evals.append(dict(np.load(base_lstm_path.format(n, h), allow_pickle=True)))
#         lstm_labels.append(f"LSTM {n}x{h}")

evaluations_to_plot = [  # (evaluation, label)
    (eval_cv, "CV"),

    (eval_lstm_3x128, "LSTM"),
    (eval_lstm_3x128_no_past, "LSTM no past"),
    # (eval_motionCNN, "MotionCNN"),

    # (eval_motionCNN_no_road, "MotionCNN no road"),
    # (eval_motionCNN_no_road_retrained, "MotionCNN no road retrained"),
    # (eval_motionCNN_no_road_retrained_unperturbed, "MotionCNN no road retrained unperturbed"),

    # (eval_motionCNN_no_past, "MotionCNN no past"),
    # (eval_motionCNN_no_past_retrained, "MotionCNN no past retrained"),
    # (eval_motionCNN_no_past_retrained_unperturbed, "MotionCNN no past retrained unperturbed"),
    #
    # (eval_motionCNN_noisy_heading, "MotionCNN noisy heading"),
    # (eval_motionCNN_noisy_heading_retrained, "MotionCNN noisy heading retrained"),
    # (eval_motionCNN_noisy_heading_retrained_unperturbed, "MotionCNN noisy heading retrained unperturbed"),

    # (eval_multipathPP, "MultiPath++"),
    # (eval_multipathPP_no_road, "MultiPath++ no road"),
    # (eval_multipathPP_no_past, "MultiPath++ no past"),
    # (eval_multipathPP_noisy_heading, "MultiPath++ noisy heading"),
]
# evaluations_to_plot = list(zip(lstm_evals, lstm_labels)) + [evaluations_to_plot[0]]
# evaluations_to_plot = list(zip(lstm_evals, lstm_labels))

t = np.arange(0.1, 8.01, 0.1)

# print(dict(eval_motionCNN))
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i, agent in enumerate(eval_motionCNN.keys()):

    for eval, label in evaluations_to_plot:
        eval[agent] = eval[agent].item()
        # if label.split(" ")[-1] not in ["3x128", "1x128", "CV"]:
        #     print("skip ", label)
        #     continue

        axs[i].plot(t, eval[agent]["minADE"], label=label)

    axs[i].set_title(agent)
    axs[i].set_xlabel("Time [s]")
    axs[i].set_ylabel(f"minADE [m] ({eval[agent]['count']} agents)")

    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()
