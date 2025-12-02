import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: choose a nicer seaborn style
sns.set(style="whitegrid", context="talk")

languages = ["en"]

for lang in languages:
    filename = f"training_log_{lang}_full_run.csv"
    df = pd.read_csv(filename)

    df_loss = df[df["loss"].notna()][["step", "loss"]]
    df_eval = df[df["eval_loss"].notna()][["step", "eval_loss"]]

    # Prepare long-form dataframe for seaborn
    df_loss["type"] = "Training Loss"
    df_eval["type"] = "Eval Loss"

    df_loss.rename(columns={"loss": "value"}, inplace=True)
    df_eval.rename(columns={"eval_loss": "value"}, inplace=True)

    df_plot = pd.concat([df_loss, df_eval], ignore_index=True)

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_plot,
        x="step",
        y="value",
        hue="type",
        linewidth=2.2
    )
    plt.legend(title="")

    plt.title(f"Loss Curves for English Decoder-only Fine-Tuning (7k steps)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("plot_lr_en_dec.png", dpi=300, bbox_inches="tight")
    plt.show()
