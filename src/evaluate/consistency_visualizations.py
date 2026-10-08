"""Create four consistency figures from per-image and mean CSVs.

Run: python create_visualizations.py [LOG_FOLDER]
Outputs are written to LOG_FOLDER/visualizations. Requires numpy and matplotlib.
"""
import argparse
import shlex
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent
LABEL = SOURCE.name.removeprefix("consistency_results_").replace("_", " ")
QUANTILES = (0.7, 0.8, 0.9)
THRESHOLD = 0.8
# LabelMapping retains background at index 0; Cityscapes training IDs shift by 1.
CLASSES = {12: "Person", 13: "Rider", 14: "Car", 15: "Truck", 16: "Bus"}
# One-based order from src/data/dataset/panoptic_parts.py:CITYSCAPES_PARTS.
PARTS = {12: ("Torso", "Head", "Arm", "Leg"),
         13: ("Torso", "Head", "Arm", "Leg"),
         **{c: ("Window", "Wheel", "Light", "License plate", "Chassis") for c in (14, 15, 16)}}
QCOLORS = ("#258579", "#326bb0", "#a85b27")
CCOLORS = {12: "#326bb0", 13: "#8053a5", 14: "#258579", 15: "#a85b27", 16: "#aa4c70"}


def read_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "class_id" in rows[0]:
        return rows
    # Native ProtoSeg logs store one column per part; blanks mean unavailable.
    class_ids = {name.lower(): c for c, name in CLASSES.items()}
    normalized = []
    for row in rows:
        for column, value in row.items():
            if not column.startswith("part_") or value.strip().lower() in ("", "nan"):
                continue
            part = int(column.removeprefix("part_"))
            assert part > 0, f"Unexpected populated background part: {path}"
            record = dict(class_id=class_ids[row["class"].lower()],
                          prototype_id=int(row["proto_id"]), part_id=part)
            if "img_id" in row:
                assert float(value) in (0, 1)
                record.update(image_index=row["img_id"], present=str(bool(float(value))))
            else:
                assert float(row["is_consistent"]) in (0, 1)
                record.update(mean_presence=value,
                              is_consistent=str(bool(float(row["is_consistent"]))))
            normalized.append(record)
    return normalized


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data():
    values, supports, prototypes, stats = {}, {}, {}, {}
    for q in QUANTILES:
        sums, images, seen = Counter(), defaultdict(set), set()
        for r in read_csv(SOURCE / f"part_presence_th_0.8_qt_{q}.csv"):
            key = int(r["class_id"]), int(r["prototype_id"]), int(r["part_id"])
            c, proto, part = key
            assert c in CLASSES and 1 <= part <= len(PARTS[c])
            assert r["present"] in ("True", "False")
            observation = key + (r["image_index"],)
            assert observation not in seen, f"Duplicate observation: {observation}"
            seen.add(observation)
            sums[key] += int(r["present"] == "True")
            images[key].add(r["image_index"])
        values[q] = {}
        mean_flags = {}
        for r in read_csv(SOURCE / f"part_presence_mean_th_0.8_qt_{q}.csv"):
            key = int(r["class_id"]), int(r["prototype_id"]), int(r["part_id"])
            assert key not in values[q]
            mean = float(r["mean_presence"])
            if "num_images" in r:
                assert len(images[key]) == int(r["num_images"])
            assert abs(mean - sums[key] / len(images[key])) < 1e-12
            values[q][key] = mean
            mean_flags[key] = r["is_consistent"] == "True"
        assert set(values[q]) == set(images)
        current_prototypes = {c: sorted({p for cc, p, _ in values[q] if cc == c}) for c in CLASSES}
        if prototypes:
            assert prototypes == current_prototypes
        prototypes = current_prototypes
        supports[q] = {}
        for c in CLASSES:
            for part in range(1, len(PARTS[c]) + 1):
                sets = [images[c, p, part] for p in prototypes[c]]
                assert sets and all(x == sets[0] for x in sets)
                supports[q][c, part] = frozenset(sets[0])
        summary_path = SOURCE / f"consistency_summary_th_0.8_qt_{q}.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if summary is not None:
            assert summary["consistency_threshold"] == THRESHOLD
            assert summary["activation_quantile"] == q
            json_protos = {(r["class_id"], r["prototype_id"]): r for r in summary["prototype_consistency"]}
            assert set(json_protos) == {(c, p) for c in CLASSES for p in prototypes[c]}
        stats[q] = {}
        for c in CLASSES:
            passing = 0
            for proto in prototypes[c]:
                best = max(v for (cc, p, _), v in values[q].items() if (cc, p) == (c, proto))
                consistent = best > THRESHOLD
                if summary is not None:
                    assert abs(best - json_protos[c, proto]["max_part_consistency"]) < 1e-12
                    assert consistent == json_protos[c, proto]["consistent"]
                assert all(flag == consistent for (cc, p, _), flag in mean_flags.items() if (cc, p) == (c, proto))
                passing += consistent
            stats[q][c] = passing, len(prototypes[c])
        overall = sum(x[0] for x in stats[q].values()) / sum(x[1] for x in stats[q].values())
        if summary is not None:
            assert abs(overall - summary["score"]) < 1e-12
        assert abs(overall - float((SOURCE / f"consistency_score_th_0.8_qt_{q}.txt").read_text())) < 1e-12
    assert all(supports[q] == supports[QUANTILES[0]] for q in QUANTILES)
    return values, supports[QUANTILES[0]], prototypes, stats


def save(fig, name, pdf):
    fig.savefig(OUT / f"{name}.png", dpi=170, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def heatmaps(values, prototypes, pdf):
    fig, axes = plt.subplots(5, 3, figsize=(16, max(17, 12 + .2 * sum(map(len, prototypes.values())))),
                             gridspec_kw={"height_ratios": [len(prototypes[c]) + 1 for c in CLASSES]})
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#d5d5d5")
    records = []
    for i, c in enumerate(CLASSES):
        for j, q in enumerate(QUANTILES):
            ax = axes[i, j]
            data = np.array([[values[q].get((c, p, part), np.nan) for part in range(1, len(PARTS[c]) + 1)]
                             for p in prototypes[c]])
            ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="none")
            ax.set_xticks(range(len(PARTS[c])), [s.replace(" ", "\n") for s in PARTS[c]], fontsize=9)
            ax.set_yticks(range(len(prototypes[c])), [f"P{p}" for p in prototypes[c]], fontsize=9)
            ax.tick_params(length=0)
            ax.set_title(f"{CLASSES[c]}  ·  quantile {q}", fontsize=12, loc="left", pad=10, weight="bold")
            for row, proto in enumerate(prototypes[c]):
                for col, part in enumerate(range(1, len(PARTS[c]) + 1)):
                    v = data[row, col]
                    ax.text(col, row, f"{v:.0%}" if np.isfinite(v) else "—", ha="center", va="center",
                            fontsize=9, color="white" if v > .58 else "#16212d")
                    if v > THRESHOLD:
                        ax.add_patch(Rectangle((col - .45, row - .45), .9, .9, fill=False, edgecolor="#f2a900", linewidth=2.4))
                    records.append(dict(quantile=q, class_name=CLASSES[c], prototype_id=proto,
                                        part_id=part, part_name=PARTS[c][part - 1], mean_presence=v,
                                        above_threshold=bool(v > THRESHOLD)))
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("Which parts does each prototype repeatedly activate on?", fontsize=21, x=.07, ha="left", y=.983)
    fig.text(.07, .955, f"{LABEL}  ·  Rows: prototypes  ·  Cells: activation frequency among images where the part is available", fontsize=11)
    fig.subplots_adjust(left=.07, right=.93, top=.925, bottom=.115, hspace=.77, wspace=.26)
    cax = fig.add_axes((.31, .065, .4, .012))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, orientation="horizontal", format=PercentFormatter(1))
    cb.set_label("Activation frequency", fontsize=10)
    fig.text(.5, .015, "Gold outline: exact frequency > 80%. Cell labels are rounded. Gray: unavailable.\nA prototype passes if any part exceeds 80%; several parts can pass.", ha="center", fontsize=10)
    save(fig, "01_prototype_part_heatmaps", pdf)
    write_csv("01_heatmap_values.csv", records)


def class_bars(stats, pdf):
    fig, ax = plt.subplots(figsize=(12, 6.8))
    xs = np.arange(len(CLASSES))
    records = []
    for j, q in enumerate(QUANTILES):
        rates = [stats[q][c][0] / stats[q][c][1] for c in CLASSES]
        bars = ax.bar(xs + (j - 1) * .24, rates, width=.22, color=QCOLORS[j], label=f"Quantile {q}", zorder=3)
        for bar, c in zip(bars, CLASSES):
            yes, n = stats[q][c]
            ax.annotate(f"{yes}/{n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=11, weight="bold")
            records.append(dict(quantile=q, class_name=CLASSES[c], consistent_prototypes=yes,
                                total_prototypes=n, consistency_fraction=yes / n))
    ax.set_xticks(xs, [f"{CLASSES[c]}\n{stats[.7][c][1]} prototypes" for c in CLASSES], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Share of prototypes that are consistent")
    ax.grid(axis="y", alpha=.2, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.suptitle("Consistency by object class", fontsize=21, x=.09, ha="left", y=.96)
    fig.text(.09, .90, "A prototype is consistent when at least one part has activation frequency strictly above 80%.", fontsize=11)
    totals = [(q, sum(a for a, _ in stats[q].values()), sum(b for _, b in stats[q].values())) for q in QUANTILES]
    fig.text(.09, .045, "Overall (each prototype counted once):  " + "    |    ".join(f"q={q}: {a}/{b} ({a/b:.1%})" for q, a, b in totals), fontsize=11)
    fig.subplots_adjust(left=.09, right=.97, top=.84, bottom=.18)
    save(fig, "02_consistency_by_class", pdf)
    write_csv("02_class_consistency.csv", records)


def prototype_lines(values, prototypes, pdf):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    records = []
    for ax, c in zip(axes.flat, CLASSES):
        for idx, proto in enumerate(prototypes[c]):
            scores, best_parts = [], []
            for q in QUANTILES:
                ps = {part: values[q][c, proto, part] for part in range(1, len(PARTS[c]) + 1)}
                best = max(ps.values())
                names = [PARTS[c][part - 1] for part, val in ps.items() if abs(val - best) < 1e-12]
                scores.append(best)
                records.append(dict(quantile=q, class_name=CLASSES[c], prototype_id=proto,
                                    max_part_frequency=best, best_parts="; ".join(names), consistent=best > THRESHOLD))
            ax.plot(QUANTILES, scores, marker="o", markersize=6, linewidth=1.9,
                    color=plt.get_cmap("tab10")(idx), linestyle="-" if idx < 5 else "--", label=f"P{proto}")
        ax.axhline(THRESHOLD, color="#595959", linestyle=(0, (3, 3)), linewidth=1.3, zorder=0)
        ax.set_title(f"{CLASSES[c]} · {len(prototypes[c])} prototypes", loc="left", fontsize=13, weight="bold")
        ax.set_xlim(.685, .915)
        ax.set_ylim(0, 1.03)
        ax.set_xticks(QUANTILES)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Activation quantile")
        ax.set_ylabel("Highest part activation frequency")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.21), ncols=5 if len(prototypes[c]) > 6 else 3, frameon=False, fontsize=9)
    axes[2, 1].axis("off")
    axes[2, 1].text(.05, .88, "Reading this plot", weight="bold", fontsize=15, va="top")
    axes[2, 1].text(.05, .74, "One line = one prototype.\n\nEach point is its strongest part score\nat that quantile. The strongest part\ncan change between points.\n\nDashed gray line = 80% cutoff.\nPoints must be above it to pass.\n\nHigher quantiles retain fewer activated\npixels. Lines connect the three evaluated\nsettings; they are not training curves.", fontsize=12, linespacing=1.5, va="top")
    fig.suptitle("Which prototypes stay consistent at stricter quantiles?", fontsize=21, x=.07, ha="left", y=.978)
    fig.text(.07, .945, f"{LABEL}  ·  Maximum part frequency for each of the {sum(map(len, prototypes.values()))} evaluated prototypes", fontsize=11)
    fig.subplots_adjust(left=.08, right=.97, top=.90, bottom=.10, hspace=.75, wspace=.23)
    save(fig, "03_prototype_scores_across_quantiles", pdf)
    write_csv("03_prototype_scores.csv", records)


def support_bars(supports, prototypes, pdf):
    fig, ax = plt.subplots(figsize=(12, 11))
    positions, labels, numbers, colors, records = [], [], [], [], []
    y = 0
    for c in CLASSES:
        for part, name in enumerate(PARTS[c], 1):
            n = len(supports[c, part])
            positions.append(y)
            labels.append(f"{CLASSES[c]} / {name}")
            numbers.append(n)
            colors.append(CCOLORS[c])
            records.append(dict(class_name=CLASSES[c], part_id=part, part_name=name,
                                usable_images=n, evaluated_prototypes=len(prototypes[c])))
            y += 1
        y += .65
    bars = ax.barh(positions, numbers, color=colors, height=.73, zorder=3)
    ax.bar_label(bars, padding=6, fontsize=10)
    ax.set_yticks(positions, labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(numbers) * 1.13)
    ax.set_xlabel("Unique images with an available part annotation", fontsize=11, labelpad=12)
    ax.grid(axis="x", alpha=.2, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("How many images support each part score?", fontsize=21, x=.08, ha="left", y=.975)
    fig.text(.08, .937, f"{LABEL}  ·  The available image sets are identical at all three quantiles", fontsize=11)
    fig.text(.08, .018, "Counts include both activated and non-activated observations. Each image is counted once per class–part pair,\nregardless of how many prototypes were evaluated. These are the denominators used for the part frequencies.", fontsize=10)
    fig.subplots_adjust(left=.24, right=.97, top=.90, bottom=.14)
    save(fig, "04_usable_images_by_part", pdf)
    write_csv("04_usable_images.csv", records)


def main(argv=None):
    global SOURCE, OUT, LABEL, PARTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_folder", nargs="?", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Log directory (default: the script's parent log directory).",
    )
    args = parser.parse_args(argv)
    SOURCE = args.log_folder.expanduser().resolve()
    if not SOURCE.is_dir():
        parser.error(f"Log directory does not exist: {SOURCE}")
    for q in QUANTILES:
        for pattern in ("part_presence_th_0.8_qt_{}.csv",
                        "part_presence_mean_th_0.8_qt_{}.csv",
                        "consistency_score_th_0.8_qt_{}.txt"):
            path = SOURCE / pattern.format(q)
            if not path.is_file():
                parser.error(f"Missing required log file: {path}")
    OUT = SOURCE / "visualizations"
    LABEL = SOURCE.name.removeprefix("consistency_results_").replace("_", " ")
    # Include additional observed IDs without inventing anatomical names.
    official_parts = {12: ("Torso", "Head", "Arm", "Leg"),
                      13: ("Torso", "Head", "Arm", "Leg"),
                      **{c: ("Window", "Wheel", "Light", "License plate", "Chassis")
                         for c in (14, 15, 16)}}
    PARTS = dict(official_parts)
    for q in QUANTILES:
        for row in read_csv(SOURCE / f"part_presence_mean_th_0.8_qt_{q}.csv"):
            c, part = int(row["class_id"]), int(row["part_id"])
            if c not in PARTS or part < 1:
                parser.error(f"Unsupported class/part ID: {c}/{part}")
            if part > len(PARTS[c]):
                PARTS[c] += tuple(f"Part {i} (unmapped)" for i in range(len(PARTS[c]) + 1, part + 1))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    values, supports, prototypes, stats = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT / "all_four_visualizations.pdf") as pdf:
        heatmaps(values, prototypes, pdf)
        class_bars(stats, pdf)
        prototype_lines(values, prototypes, pdf)
        support_bars(supports, prototypes, pdf)
    (OUT / "README.md").write_text(
        f"# Consistency visualizations: {LABEL}\n\n"
        "The four figures are available as individual PNGs/PDFs and in all_four_visualizations.pdf. "
        "Each figure also has a CSV with exact values.\n\n"
        "1. 01_prototype_part_heatmaps: per-prototype part frequencies, with classes in rows and "
        "quantiles in columns. Gold outlines mark exact values strictly above 0.8. Displayed percentages are rounded.\n"
        "2. 02_consistency_by_class: consistent prototypes divided by all evaluated prototypes of each class. "
        "Labels show passing/total counts. Overall score weights prototypes equally, not classes equally.\n"
        "3. 03_prototype_scores_across_quantiles: each prototype's highest part frequency at each quantile. "
        "The strongest part may change; the corresponding part names are in the CSV. Gray dashed line is 0.8.\n"
        "4. 04_usable_images_by_part: unique image_index (or ProtoSeg img_id) counts per class and part, including True and False. "
        "Available image sets match across prototypes within each class/part and across all quantiles, "
        "so a single support chart covers the three settings. Do not sum counts across parts to obtain unique dataset images.\n\n"
        "Sources: ../part_presence_th_0.8_qt_*.csv, ../part_presence_mean_th_0.8_qt_*.csv, "
        "../consistency_summary_th_0.8_qt_*.json, ../consistency_score_th_0.8_qt_*.txt. "
        "Counts and averages are recomputed from per-image observations and checked against saved means, "
        "consistency flags, overall scores, and JSON summaries when available. Missing annotations are excluded. "
        "All evaluated prototypes are included; no filter on consistency is applied.\n\n"
        "Class IDs 12–16 are person/rider/car/truck/bus, following src/data/dataset/label_mapping.py "
        "(background retained at index 0). Part IDs use src/data/dataset/panoptic_parts.py:CITYSCAPES_PARTS.\n\n"
        "Additional observed part IDs are labeled unmapped and included in the calculations.\n\n"
        f"Regenerate: `python {shlex.quote(str(Path(__file__).resolve()))} {shlex.quote(str(SOURCE))}` "
        "(requires numpy and matplotlib).\n"
    )
    print(f"Validated {sum(map(len, prototypes.values()))} prototypes and {len(supports)} class–part pairs across 3 quantiles.")
    print(f"Output directory: {OUT}")
    print("Saved 4 PNGs, 4 individual PDFs, a combined 4-page PDF, 4 data CSVs, and README.md.")
    for q in QUANTILES:
        print(q, {CLASSES[c]: f"{a}/{b}" for c, (a, b) in stats[q].items()})


if __name__ == "__main__":
    main()
