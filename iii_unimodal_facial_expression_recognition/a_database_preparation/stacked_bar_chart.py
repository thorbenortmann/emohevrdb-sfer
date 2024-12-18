from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Mapping from label IDs to emotion names
EMOTION_LABELS = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Neutral',
    5: 'Sadness',
    6: 'Surprise'
}


def create_dataframe(training_csv, validation_csv, test_csv):
    """Creates a DataFrame with the samples per emotion and set."""
    # Load each CSV into a DataFrame, selecting only the 'timestamp' and 'label' columns
    training_df = pd.read_csv(training_csv, usecols=['timestamp', 'Label'])
    validation_df = pd.read_csv(validation_csv, usecols=['timestamp', 'Label'])
    test_df = pd.read_csv(test_csv, usecols=['timestamp', 'Label'])

    # Add a column to each DataFrame to indicate the set type
    training_df['Set'] = 'training_set'
    validation_df['Set'] = 'validation_set'
    test_df['Set'] = 'test_set'

    # Concatenate all DataFrames
    combined_df = pd.concat([training_df, validation_df, test_df], ignore_index=True)

    # Map label IDs to emotion names
    combined_df['Emotion'] = combined_df['Label'].map(EMOTION_LABELS)

    # Group by Emotion and Set to get the count of samples per emotion
    records = combined_df.groupby(['Emotion', 'Set']).size().reset_index(name='Samples')

    return records


def plot_stacked_barchart(data):
    """Creates a stacked bar chart from the DataFrame."""
    # Pivot the DataFrame to prepare it for plotting
    data_pivoted = data.pivot(index='Emotion', columns='Set', values='Samples').fillna(0)

    # Adjust the order of the sets for plotting
    data_pivoted = data_pivoted[['test_set', 'validation_set', 'training_set']]

    # Define the color scheme, reverse the order so dark blue is at the bottom
    color_palette = sns.color_palette("Blues_d", n_colors=len(data_pivoted.columns))[::-1]

    # Create the bar chart
    ax = data_pivoted.plot(kind='bar', stacked=True, figsize=(8, 4), width=0.6, color=color_palette)

    # Annotate each bar with its value
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:  # Only annotate if the bar is visible
            ax.text(x + width / 2,
                    y + height / 2,
                    str(int(height)),
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='white',
                    fontsize=11)

    # Write the total sum of segments per emotion above the bars
    for i, (index, row) in enumerate(data_pivoted.iterrows()):
        total_height = row.sum()
        ax.text(i, total_height + 8, f'{int(total_height)}',
                ha='center', va='bottom', fontsize=12)

    plt.xlabel('')
    plt.ylabel('Number of Samples', fontsize=12,
               fontweight='bold'
               )
    plt.xticks(fontsize=13, rotation=0, fontweight='bold')
    plt.yticks(fontsize=13)

    # Set Y-axis labels in steps of 100
    ax.set_yticks(range(0, max(data_pivoted.sum(axis=1)) + 1, 100))

    plt.legend(title='', labels=['Training', 'Validation', 'Test'], fontsize=11,
               bbox_to_anchor=(0.0, 1.05), loc='upper left')

    # Remove Y-axis gridlines to further clean up the chart
    ax.yaxis.grid(False)

    # Only remove the top and right frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(-0.4, 6.4)

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / 'stacked_bar_chart.png'), dpi=400)


if __name__ == '__main__':
    base_path = Path(r'/media/thor/PortableSSD/emoji-hero-vr-db/dataset/emoji-hero-vr-db/emoji-hero-vr-db-sfea-as-csv')
    training_csv_path = str(base_path / 'training_set.csv')
    validation_csv_path = str(base_path / 'validation_set.csv')
    test_csv_path = str(base_path / 'test_set.csv')

    # Create DataFrame and plot the chart
    data = create_dataframe(training_csv_path, validation_csv_path, test_csv_path)
    plot_stacked_barchart(data)
