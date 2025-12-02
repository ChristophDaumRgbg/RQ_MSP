import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import evaluate

LANGUAGES = ["en", "ja", "pa", "ur", "vi", "yo", "de", "ar", "gl", "it", "sw", "ta", "zh", "ru"]

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "pa": "Punjabi",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "yo": "Yoruba",
    "de": "German",
    "ar": "Arabic",
    "gl": "Galician",
    "it": "Italian",
    "sw": "Swahili",
    "ta": "Tamil",
    "zh": "Chinese",
    "ru": "Russian"
}


SPLIT_BY_NOISE_LEVEL = True     # additional noise-level plots ON/OFF
BASE_DIR = "files"

FNAME_BASE = "base_whisper_results.csv"
FNAME_EN_ADAPTER = "lora_en_whisper_results.csv"
FNAME_LANG_ADAPTER = "lora_{lang}_whisper_results.csv"

OUTPUT_CSV = "evaluation_summary.csv"

PLOT_WER = "plot_wer.png"
PLOT_CER = "plot_cer.png"
PLOT_WER_NOISE = "plot_wer_noise.png"
PLOT_CER_NOISE = "plot_cer_noise.png"

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(df: pd.DataFrame):
    preds = df["prediction"].fillna("").apply(lambda x: x if isinstance(x, str) else "").tolist()

    refs = df["reference"].fillna("").apply(lambda x: x if isinstance(x, str) else "").tolist()


    return {
        "WER": wer_metric.compute(predictions=preds, references=refs),
        "CER": cer_metric.compute(predictions=preds, references=refs)
    }

def load_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return None
    return pd.read_csv(path)


def load_language_files(lang: str):
    folder = os.path.join(BASE_DIR, lang)
    files = {
        "base": os.path.join(folder, FNAME_BASE),
        "adapter": os.path.join(folder, FNAME_LANG_ADAPTER.format(lang=lang)),
        "decoder": os.path.join(folder, f"lora_{lang}_whisper_results_decoder.csv"),
        "encoder": os.path.join(folder, f"lora_{lang}_whisper_results_encoder.csv"),
        "decoder_7k": os.path.join(folder, f"lora_{lang}_whisper_results_decoder_7k.csv"),
        "encoder_7k": os.path.join(folder, f"lora_{lang}_whisper_results_encoder_7k.csv"),
    }
    if lang != "en":
        files["en_adapter"] = os.path.join(folder, FNAME_EN_ADAPTER)

    return {name: load_file(path) for name, path in files.items()}

def evaluate_language(lang: str, split_noise: bool) -> pd.DataFrame:
    print(f"\n===== Evaluating language: {lang} =====")

    dfs = load_language_files(lang)
    rows = []

    for model_name, df in dfs.items():
        if df is None:
            continue

        metrics = compute_metrics(df)
        rows.append({
            "language": lang,
            "model": model_name,
            "noise_level": "ALL",
            "WER": metrics["WER"],
            "CER": metrics["CER"],
        })

        if split_noise:
            for nl, sub in df.groupby("noise_level"):
                metrics = compute_metrics(sub)
                rows.append({
                    "language": lang,
                    "model": model_name,
                    "noise_level": nl,
                    "WER": metrics["WER"],
                    "CER": metrics["CER"],
                })

    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, metric: str, output_file: str):
    missing = (df["language"] == "de") & (df["model"] == "decoder_7k")
    if missing.any():
        pass
    else:
        # duplicate decoder row and rename model to decoder_7k
        de_decoder = df[(df["language"] == "de") & (df["model"] == "decoder")].copy()
        de_decoder["model"] = "decoder_7k"
        df = pd.concat([df, de_decoder], ignore_index=True)

    data = df[df["noise_level"] == "ALL"].copy()

    model_rename = {
        "en_adapter": "English Adapter",
        "decoder_7k": "Decoder (7k)",
        "encoder_7k": "Encoder (7k)",
        "decoder": "Decoder",
        "encoder": "Encoder",
        "base": "Base",
        "adapter": "Adapter",
    }

    data["model_pretty"] = data["model"].replace(model_rename)

    hue_order = [
        "Base",
        "Adapter",
        "English Adapter",
        "Decoder",
        "Encoder",
        "Decoder (7k)",
        "Encoder (7k)",
    ]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data,
        x="language",
        y=metric,
        hue="model_pretty",
        hue_order=hue_order,
        errorbar=None,
        palette="tab10",
    )

    plt.title(f"{metric} by Language and Model")
    plt.ylabel(metric)
    plt.xlabel("Language")
    plt.legend(title="Model")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()




def plot_metric_noise(df: pd.DataFrame, metric: str, output_file: str):
    data = df[df["noise_level"] != "ALL"]

    if data.empty:
        print(f"No noise-level data for {metric}, skipping.")
        return

    label_map = {
        "base": "no adapter",
        "adapter": "language adapter",
        "en_adapter": "English adapter",
    }

    languages = sorted(data["language"].unique())
    if len(languages) != 2:
        print("Expected exactly 2 languages for this plot.")
        return

    lang1, lang2 = languages

    plt.figure(figsize=(16, 6))

    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(
        data=data[data["language"] == lang1],
        x="noise_level",
        y=metric,
        hue="model",
        palette="tab10",
        errorbar=None,
        ax=ax1
    )
    ax1.set_title(f"{LANGUAGE_NAMES.get(lang1, lang1)}")
    ax1.set_xlabel("Noise Level")
    ax1.set_ylabel(metric)

    # Rename legend labels
    handles, labels = ax1.get_legend_handles_labels()
    new_labels = [label_map[l] for l in labels]
    ax1.legend(handles, new_labels, title="Adapter Type", frameon=True)

    # --- RIGHT subplot ---
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(
        data=data[data["language"] == lang2],
        x="noise_level",
        y=metric,
        hue="model",
        palette="tab10",
        errorbar=None,
        ax=ax2
    )
    ax2.set_title(f"{LANGUAGE_NAMES.get(lang2, lang2)}")
    ax2.set_xlabel("Noise Level")
    ax2.set_ylabel(metric)

    # Match legend formatting
    handles, labels = ax2.get_legend_handles_labels()
    new_labels = [label_map[l] for l in labels]
    ax2.legend(handles, new_labels, title="Adapter Type", frameon=True)

    plt.suptitle(f"{metric} by Noise Level", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved noise-level plot: {output_file}")



def plot_per_language_all_models(df: pd.DataFrame, metric: str, output_root: str):
    out_dir = os.path.join(output_root, metric)
    os.makedirs(out_dir, exist_ok=True)

    for lang, sub in df.groupby("language"):
        lang_data = sub[sub["noise_level"] == "ALL"]

        if lang_data.empty:
            print(f"[WARN] No ALL-level data for {lang}, skipping per-language plot.")
            continue

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=lang_data,
            x="model",
            y=metric,
            errorbar=None,
            palette="tab20"
        )
        plt.title(f"{metric} for {lang}")
        plt.xlabel("Model Variant")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{lang}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved per-language plot: {out_path}")

def plot_per_language_noise(df: pd.DataFrame, metric: str, output_root: str):
    out_dir = os.path.join(output_root, metric)
    os.makedirs(out_dir, exist_ok=True)

    model_rename = {
        "en_adapter": "English Adapter",
        "decoder_7k": "Decoder (7k)",
        "encoder_7k": "Encoder (7k)",
        "decoder": "Decoder",
        "encoder": "Encoder",
        "base": "Base",
        "adapter": "Adapter",
    }

    df_noise = df[df["noise_level"] != "ALL"].copy()
    df_noise["model_pretty"] = df_noise["model"].replace(model_rename)

    hue_order = [
        "Base",
        "Adapter",
        "English Adapter",
        "Decoder",
        "Encoder",
        "Decoder (7k)",
        "Encoder (7k)",
    ]

    for lang, sub in df_noise.groupby("language"):
        if sub.empty:
            print(f"[WARN] No noise-level data for {lang}, skipping noise plot.")
            continue

        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=sub,
            x="noise_level",
            y=metric,
            hue="model_pretty",
            hue_order=[h for h in hue_order if h in sub["model_pretty"].unique()],
            errorbar=None,
            palette="tab10"
        )

        plt.title(f"{metric} by Noise Level — {LANGUAGE_NAMES.get(lang, lang)}")
        plt.xlabel("Noise Level")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Model")

        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{lang}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved per-language noise-level plot: {out_path}")

def compute_improvement_scatter_data(df: pd.DataFrame, metric: str, min_noise: float | None = None):
    tmp = df[df["noise_level"] != "ALL"].copy()

    tmp["noise_numeric"] = tmp["noise_level"].astype(float)

    if min_noise is not None:
        tmp = tmp[tmp["noise_numeric"] >= min_noise]

    pivot = tmp.pivot_table(
        index=["language", "noise_level"],
        columns="model",
        values=metric
    )

    if "base" not in pivot.columns or "adapter" not in pivot.columns:
        raise ValueError(f"Missing 'base' or 'adapter' columns for {metric}")

    pivot["imp_lang"] = pivot["base"] - pivot["adapter"]

    if "en_adapter" in pivot.columns:
        pivot["imp_en"] = pivot["base"] - pivot["en_adapter"]
    else:
        pivot["imp_en"] = np.nan

    out = pivot[["imp_lang", "imp_en"]].reset_index()

    return out



def plot_scatter_combined_metrics(df: pd.DataFrame, out_file: str):
    rows = []

    for metric in ["WER", "CER"]:
        data = compute_improvement_scatter_data(df, metric, min_noise=None)

        mean_data = data.groupby("language")[["imp_lang", "imp_en"]].mean().reset_index()
        mean_data["Language"] = mean_data["language"].map(LANGUAGE_NAMES)
        mean_data["Metric"] = metric

        rows.append(mean_data)

    merged = pd.concat(rows, ignore_index=True)

    plt.figure(figsize=(10, 10))
    palette = sns.color_palette("tab20", n_colors=len(LANGUAGE_NAMES))

    ax = sns.scatterplot(
        data=merged,
        x="imp_lang",
        y="imp_en",
        hue="Language",
        style="Metric",
        s=150,
        palette=palette,
        edgecolor="black",  # black border
        linewidth=0.6  # thinner border
    )

    # crosshairs at "no improvement"
    plt.axhline(0, color="black", linewidth=1, alpha=0.6)
    plt.axvline(0, color="black", linewidth=1, alpha=0.6)

    # crop tightly around data
    pad = 0.05
    xmin, xmax = merged["imp_lang"].min(), merged["imp_lang"].max()
    ymin, ymax = merged["imp_en"].min(), merged["imp_en"].max()
    xr = xmax - xmin if xmax != xmin else 0.1
    yr = ymax - ymin if ymax != ymin else 0.1
    plt.xlim(xmin - xr * pad, xmax + xr * pad)
    plt.ylim(ymin - yr * pad, ymax + yr * pad)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("Language Adapter", fontsize=12)
    plt.ylabel("English Adapter", fontsize=12)
    plt.title("Relative WER and CER Improvements after Fine-Tuning", fontsize=14)

    plt.legend(
        title=None,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")

def main():
    sns.set_style("whitegrid")
    if os.path.exists(OUTPUT_CSV):
        print(f"Loading cached metrics from {OUTPUT_CSV} ...")
        final = pd.read_csv(OUTPUT_CSV)
    else:
        print("No metrics cache found — running full evaluation...")
        all_results = []

        for lang in LANGUAGES:
            df = evaluate_language(lang, SPLIT_BY_NOISE_LEVEL)
            if df is not None and not df.empty:
                all_results.append(df)

        if not all_results:
            print("No results computed.")
            return

        final = pd.concat(all_results, ignore_index=True)

        print("\n===== FINAL RESULTS =====")
        print(final)

        final.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved metrics CSV to {OUTPUT_CSV}")

    # Regular plots (always produced)
    plot_metric(final, "WER", PLOT_WER)
    plot_metric(final, "CER", PLOT_CER)

    # Noise-level plots
    if SPLIT_BY_NOISE_LEVEL:
        plot_metric_noise(final, "WER", PLOT_WER_NOISE)
        plot_metric_noise(final, "CER", PLOT_CER_NOISE)

    # Per-language all-model plots
    plot_per_language_all_models(final, "WER", "per_language_plots")
    plot_per_language_all_models(final, "CER", "per_language_plots")

    #Per-language noise-level plots
    plot_per_language_noise(final, "WER", "per_language_noise_plots")
    plot_per_language_noise(final, "CER", "per_language_noise_plots")

    # Scatter combined noise, WER & CER
    plot_scatter_combined_metrics(final, out_file="scatter_combined_WER_CER.png")


if __name__ == "__main__":
    main()
