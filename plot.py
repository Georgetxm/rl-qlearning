import time
from datetime import datetime
import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores, is_final=False, file_name=None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    # Save the plot to a file
    if is_final:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        if file_name:
            plt.savefig(f"./plots/{file_name}_{formatted_time}.png")
        else:
            plt.savefig(f"./plots/{formatted_time}.png")
