from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_plot_wordcloud(dataset, title):
    plt.figure(figsize = (20, 20))
    plt.axis('off')
    plt.imshow(WordCloud(background_color="white", mode="RGBA").generate(dataset))
    plt.title(title)

    generate_plot_wordcloud(spam_string, 'Spam Word Clould')

    generate_plot_wordcloud(ham_string, 'Ham Word Clould')

    