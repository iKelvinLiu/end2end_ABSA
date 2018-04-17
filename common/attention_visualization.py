import time
import numpy as np
from matplotlib import pyplot as plt
import codecs


def create_attention_figure(source_words, target_words, source_len, target_len, attention_score, color):
    # Plot the figure with sentence length
    if str(color) == "True":
        color_map = plt.cm.Blues
    else:
        color_map = plt.cm.Reds
    fig = plt.figure(figsize=(25, 10))
    plt.imshow(
          X=attention_score[:target_len, :source_len],
          interpolation="nearest",
          cmap=color_map,
          vmin=0,
          vmax=0.2)
    plt.colorbar(shrink=.30)
    plt.xticks(np.arange(source_len), source_words, rotation=60, fontsize=18)
    plt.yticks(np.arange(target_len), target_words, fontsize=18)

    fig.tight_layout()
    return fig


def batch_plot_attention(hop, attentions, source_file, index, epoch, path, predict):
    fp = codecs.open(source_file, 'r', 'utf-8')
    lines = fp.readlines()
    for i in index:
        aspect_word = [' '.join(lines[i*3 + 1].lower().split())]
        sentence = lines[i*3].split()
        sentence = ['\\$Target\\$' if x == '$T$' else x for x in sentence]
        create_attention_figure(source_words=sentence,
                                target_words=aspect_word,
                                source_len=len(sentence),
                                target_len=len(aspect_word),
                                attention_score=np.expand_dims(attentions[hop][0][i], axis=0),
                                color=predict[i],
                                )
        save_name = path + "//attention_plot" + "_epoch" + str(epoch) + "_hop" + str(hop) + "_sample" + str(i)
        plt.savefig(save_name)
        plt.close()


