"""
Store Layout Analyzer - base project
------------------------------------
1. Defines a store as a grid of zones, each assigned a product category.
2. Simulates customer movement + purchases (replace with real data later).
3. Analyzes footfall, dwell time, dead zones and category affinity.
4. Prints layout recommendations and saves heatmaps.

Run:  python store_layout_analyzer.py
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# 1. STORE DEFINITION  (8 rows x 10 columns of zones)
# ----------------------------------------------------------------------------
LAYOUT = [
    ["Produce",   "Produce",   "Bakery",    "Bakery",    "Dairy",     "Dairy",     "Frozen",    "Frozen",    "Snacks",    "Snacks"],
    ["Produce",   "Produce",   "Bakery",    "Bakery",    "Dairy",     "Dairy",     "Frozen",    "Frozen",    "Snacks",    "Snacks"],
    ["Beverages", "Beverages", "Beverages", "Household", "Household", "Household", "Personal",  "Personal",  "Personal",  "Snacks"],
    ["Beverages", "Beverages", "Beverages", "Household", "Household", "Household", "Personal",  "Personal",  "Personal",  "Electronics"],
    ["Beverages", "Beverages", "Household", "Household", "Household", "Personal",  "Personal",  "Electronics", "Electronics", "Electronics"],
    ["Bakery",    "Bakery",    "Household", "Household", "Personal",  "Personal",  "Electronics", "Electronics", "Electronics", "Electronics"],
    ["Dairy",     "Dairy",     "Snacks",    "Snacks",    "Snacks",    "Frozen",    "Frozen",    "Frozen",    "Electronics", "Electronics"],
    ["Entrance",  "Entrance",  "Checkout",  "Checkout",  "Checkout",  "Checkout",  "Checkout",  "Checkout",  "Checkout",  "Checkout"],
]
ROWS, COLS = len(LAYOUT), len(LAYOUT[0])
ENTRANCE = (ROWS - 1, 0)
CHECKOUT = (ROWS - 1, 4)

# popularity = chance a shopper wants the category; margin = profit share
CATEGORY_INFO = pd.DataFrame(
    {
        "popularity": [0.70, 0.60, 0.45, 0.35, 0.40, 0.50, 0.40, 0.30],
        "margin":     [0.10, 0.12, 0.25, 0.22, 0.35, 0.18, 0.30, 0.20],
        "avg_price":  [6.0,  4.0,  5.0,  8.0,  3.5,  4.5,  9.0,  60.0],
    },
    index=["Produce", "Dairy", "Bakery", "Frozen", "Snacks", "Beverages", "Personal", "Electronics"],
)
CATEGORY_INFO.loc["Household"] = [0.45, 0.28, 12.0]

CELLS_BY_CAT = {}
for r, c in itertools.product(range(ROWS), range(COLS)):
    CELLS_BY_CAT.setdefault(LAYOUT[r][c], []).append((r, c))


# ----------------------------------------------------------------------------
# 2. SIMULATE CUSTOMER DATA  (swap this with real sensor / CCTV / POS data)
# ----------------------------------------------------------------------------
def walk(start, end):
    """Manhattan walk from start to end cell with random step order."""
    path, (r, c) = [], start
    while (r, c) != end:
        move_row = r != end[0] and (c == end[1] or rng.random() < 0.5)
        r += int(np.sign(end[0] - r)) if move_row else 0
        c += 0 if move_row else int(np.sign(end[1] - c))
        path.append((r, c))
    return path


def simulate(n_customers=1500):
    visits, baskets = [], []
    cats = CATEGORY_INFO.index.to_list()
    for cid in range(n_customers):
        wanted = [k for k in cats if rng.random() < CATEGORY_INFO.loc[k, "popularity"] * 0.6]
        if not wanted:
            wanted = [rng.choice(cats)]
        rng.shuffle(wanted)

        pos = ENTRANCE
        visits.append((cid, *pos, 2))
        for cat in wanted:
            target = CELLS_BY_CAT[cat][rng.integers(len(CELLS_BY_CAT[cat]))]
            for step in walk(pos, target):
                visits.append((cid, *step, 2))              # walking = short dwell
            visits.append((cid, *target, int(rng.integers(20, 90))))  # browsing
            pos = target
            if rng.random() < 0.85:                          # bought it
                baskets.append((cid, cat, CATEGORY_INFO.loc[cat, "avg_price"]))
        for step in walk(pos, CHECKOUT):
            visits.append((cid, *step, 2))
        visits.append((cid, *CHECKOUT, int(rng.integers(30, 120))))

    visits = pd.DataFrame(visits, columns=["customer_id", "row", "col", "dwell_sec"])
    baskets = pd.DataFrame(baskets, columns=["customer_id", "category", "spend"])
    return visits, baskets


# ----------------------------------------------------------------------------
# 3. ANALYSIS
# ----------------------------------------------------------------------------
def build_grids(visits):
    footfall = np.zeros((ROWS, COLS))
    dwell = np.zeros((ROWS, COLS))
    uniq = visits.drop_duplicates(["customer_id", "row", "col"]).groupby(["row", "col"]).size()
    for (r, c), v in uniq.items():
        footfall[r, c] = v
    for (r, c), v in visits.groupby(["row", "col"])["dwell_sec"].sum().items():
        dwell[r, c] = v
    return footfall, dwell


def category_summary(footfall, dwell, baskets):
    rows = []
    for cat, cells in CELLS_BY_CAT.items():
        rows.append(
            {
                "category": cat,
                "zones": len(cells),
                "avg_footfall": np.mean([footfall[r, c] for r, c in cells]),
                "avg_dwell_sec": np.mean([dwell[r, c] for r, c in cells]),
            }
        )
    df = pd.DataFrame(rows).set_index("category")
    sales = baskets.groupby("category")["spend"].sum().rename("revenue")
    df = df.join(sales).join(CATEGORY_INFO["margin"])
    df["profit"] = df["revenue"] * df["margin"]
    return df.fillna(0).sort_values("avg_footfall", ascending=False)


def affinity(baskets, top=5):
    """Lift of category pairs bought by the same customer."""
    onehot = pd.crosstab(baskets["customer_id"], baskets["category"]).clip(upper=1)
    n = len(onehot)
    out = []
    for a, b in itertools.combinations(onehot.columns, 2):
        both = (onehot[a] & onehot[b]).sum() / n
        lift = both / (onehot[a].mean() * onehot[b].mean())
        out.append((a, b, round(both, 3), round(lift, 2)))
    df = pd.DataFrame(out, columns=["cat_a", "cat_b", "support", "lift"])
    return df.sort_values("lift", ascending=False).head(top)


def recommendations(summary, aff):
    recs = []
    skip = {"Entrance", "Checkout"}
    s = summary.drop(index=[k for k in skip if k in summary.index])
    traffic_med = s["avg_footfall"].median()
    margin_med = s["margin"].median()

    for cat, row in s.iterrows():
        if row["margin"] > margin_med and row["avg_footfall"] < traffic_med:
            recs.append(f"MOVE '{cat}' (high margin {row['margin']:.0%}) to a higher-traffic zone - it is in a cold area.")
        if row["margin"] < margin_med and row["avg_footfall"] > traffic_med * 1.2:
            recs.append(f"'{cat}' (low margin) sits in a hot zone - swap with a high-margin category.")

    cold = s.sort_values("avg_footfall").head(2).index.tolist()
    recs.append(f"DEAD ZONES: {', '.join(cold)} get the least traffic - add promos/signage or move a popular item nearby.")

    for _, r in aff.head(3).iterrows():
        recs.append(f"CROSS-MERCHANDISE '{r.cat_a}' with '{r.cat_b}' (lift {r.lift}, bought together often).")
    return recs


# ----------------------------------------------------------------------------
# 4. VISUALIZATION
# ----------------------------------------------------------------------------
def plot(footfall, dwell, path="layout_analysis.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, grid, title, cmap in [
        (axes[0], footfall, "Footfall Heatmap (visits per zone)", "YlOrRd"),
        (axes[1], dwell, "Dwell-Time Heatmap (total seconds)", "PuBu"),
    ]:
        im = ax.imshow(grid, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for r, c in itertools.product(range(ROWS), range(COLS)):
            ax.text(c, r, LAYOUT[r][c], ha="center", va="center", fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    visits, baskets = simulate(1500)
    visits.to_csv("visits.csv", index=False)
    baskets.to_csv("baskets.csv", index=False)

    footfall, dwell = build_grids(visits)
    summary = category_summary(footfall, dwell, baskets)
    aff = affinity(baskets)
    recs = recommendations(summary, aff)
    img = plot(footfall, dwell)

    pd.set_option("display.width", 120)
    print("\n=== CATEGORY SUMMARY ===")
    print(summary.round(1))
    print("\n=== TOP CATEGORY AFFINITIES ===")
    print(aff.to_string(index=False))
    print("\n=== LAYOUT RECOMMENDATIONS ===")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r}")
    print(f"\nHeatmap saved to {img}")
