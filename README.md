# Foresight — Inventory Demand Forecasting System

Foresight looks at your past sales data and tells you two things every store manager
needs to know:

1. **How much will I sell in the coming days/weeks?** (the forecast)
2. **Do I need to reorder stock, and how soon?** (the inventory alert)

It's a web app (no coding needed to use it) — you upload a sales spreadsheet, pick a
store and product, and it draws you a forecast chart, warns you about stockouts, and
lets you download a report.

---

## 1. What problem does this solve?

If you run a shop or warehouse, you constantly have to guess: *"Do I have enough stock
to last until my next delivery?"* Guess wrong and you either run out of product
(lost sales) or over-order (wasted money tied up in stock).

Foresight replaces the guess with a data-driven answer by:

- Learning the sales pattern of each product (trend, day-of-week effects, seasonal
  peaks) from your historical data.
- Predicting demand for the next days/months.
- Calculating a **reorder point** — the stock level at which you should place a new
  order, based on your delivery lead time and a safety buffer.
- Flagging **stockout risk** the moment predicted demand is about to exceed what
  you have on hand.

## 2. What can you do with it?

- **Upload any sales CSV** — the app automatically figures out which column is the
  date, which is the sales quantity, which is the store/product, etc. (You can also
  just try it with the demo dataset that ships with the project.)
- **Pick a store and a product (SKU)** from a dropdown and see its forecast.
- **Adjust settings** in the sidebar: how far ahead to forecast, delivery lead time,
  safety stock, and your current stock level.
- **Run "what-if" scenarios** — see how a future price change or a promotion would
  shift demand.
- **See why the forecast looks the way it does** — trend, weekly pattern, yearly
  (seasonal) pattern, and unusual spikes/dips ("anomalies"), explained in plain
  English, not just charts.
- **Compare two forecasting methods** (Prophet vs. a simple linear model) and see
  which one is more accurate for that product.
- **Compare multiple products side-by-side** — pick a handful of SKUs and see their
  demand, stock health, and weekly patterns on one screen.
- **Download a report** — either a CSV of the forecast numbers or a polished PDF
  summary you can hand to a manager.

## 3. How it works, in plain English

```
Your sales CSV  →  Cleaned & organised by day  →  Forecasting model (Prophet)
                                                          │
                                                          ▼
                                     Predicted demand for future days
                                                          │
                                                          ▼
                          Compared against your current stock and delivery time
                                                          │
                                                          ▼
                        Reorder point, stockout warning, downloadable report
```

Under the hood it uses [Prophet](https://facebook.github.io/prophet/), a
forecasting library built by Meta that's designed to handle trends, weekly cycles,
and yearly seasonality automatically — you don't need to know any statistics to use
it.

---

## 4. Getting started

### Prerequisites

- Python 3.10 or newer
- [Git LFS](https://git-lfs.com/) (only needed if you want the bundled demo dataset —
  see [Data storage note](#7-data-storage-note-git-lfs) below)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mohit625/Inventory_analysis.git
cd Inventory_analysis

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt
```

> **Note on Prophet:** Prophet installs a statistical modelling engine (`cmdstanpy`)
> that compiles some code the first time it runs. On most machines `pip install` just
> works, but if it fails, check the
> [official Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html)
> for platform-specific instructions.

### Running the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`). If it
doesn't open automatically, copy the URL printed in the terminal into your browser.

---

## 5. Using the app — step by step

1. **Load data.** Either drag a CSV onto the upload box at the top, or do nothing and
   the app will use the demo dataset bundled with the project.
2. **Check the column mapping** (only shown for uploaded files). The app guesses
   which of your columns is the date, sales quantity, store, product, price, and
   promotion flag. Fix any dropdown that guessed wrong, then continue — you'll see a
   preview of your data to confirm it looks right.
3. **Pick a Store and an Item/SKU** from the sidebar dropdowns.
4. **Choose a date range** to focus on (defaults to your whole history).
5. **Tune the sidebar settings:**
   - **Forecast horizon** — how many days ahead to predict.
   - **Lead time** — how many days it takes for a new order to arrive.
   - **Safety stock** — an extra buffer kept on hand in case of demand spikes or
     delivery delays.
   - **Current stock** — either let the app estimate it from recent sales, or type in
     the real number.
6. **Read the results on the main page:**
   - Executive summary cards: next-day demand, current stock, reorder point,
     lead-time demand.
   - A stockout-risk banner and a reorder banner (green = healthy, red/yellow =
     action needed).
   - The forecast chart, with a shaded band showing the range demand will most
     likely fall in.
   - A breakdown of *why* the forecast looks that way (trend / weekly pattern /
     yearly pattern / unusual spikes).
   - A comparison between two prediction methods, so you know which is more
     trustworthy for that product.
7. **Try the "Multi-SKU Comparison" section** if you want to compare several products
   at once — pick up to 15 SKUs (it's capped so it doesn't take forever on datasets
   with thousands of products) and it forecasts just those.
8. **Download your results** from the tabs at the bottom — a CSV of the full forecast,
   or a ready-to-share PDF report.

---

## 6. Expected data format

Your CSV needs at least a **date** column and a **sales** column. Everything else is
optional — the app fills in sensible defaults if they're missing.

| Column     | Required? | What it means                          | Common alternative names the app recognizes |
|------------|-----------|-----------------------------------------|-----------------------------------------------|
| `date`     | Yes       | The day of the sale                     | `transaction_date`, `order_date`, `day`, ...  |
| `sales`    | Yes       | Units sold that day                     | `quantity`, `units_sold`, `demand`, ...       |
| `store_id` | No        | Which store/location                    | `store`, `branch`, `location`, `outlet`, ...  |
| `item_id`  | No        | Which product/SKU                       | `item`, `sku`, `product_name`, ...            |
| `price`    | No        | Selling price that day                  | `unit_price`, `mrp`, `cost`, ...              |
| `promo`    | No        | Was there a promotion running? (0/1)    | `promotion`, `discount_flag`, `offer`, ...    |

If a column is missing entirely, the app assumes a single store/product, zero price,
and no promotions — so even a bare two-column `date,sales` file will work.

---

## 7. Data storage note (Git LFS)

The demo dataset (`retail_sales.csv`) bundled with this repo is large (~190MB), so
it's stored using [Git LFS](https://git-lfs.com/) instead of being committed directly.
If you clone the repo and the file looks tiny or unreadable, run:

```bash
git lfs install
git lfs pull
```

You don't need the demo dataset at all if you're only going to analyze your own data
via the upload button.

---

## 8. Project structure

```
app.py               Streamlit web app — the UI you interact with in the browser
main.py               Core logic: loading/cleaning data, running the forecast,
                       and calculating reorder points (this also runs standalone
                       from the command line, see below)
models.py             A simple linear-regression forecaster, used as a baseline
                       to compare against Prophet
sku_comparison.py     The "compare multiple products at once" screen
pdf_report.py         Builds the downloadable PDF summary report
requirements.txt      Python packages this project depends on
retail_sales.csv      Demo dataset (stored via Git LFS, see above)
.streamlit/config.toml  App theme settings
```

You can also run the forecasting logic on its own, without the web app, as a quick
command-line demo:

```bash
python3 main.py
```

This prints a forecast summary for the first store/product it finds in the dataset,
useful for testing changes without opening a browser.

---

## 9. Tech stack

- **[Streamlit](https://streamlit.io/)** — the web app framework
- **[Prophet](https://facebook.github.io/prophet/)** — the main forecasting model
- **[scikit-learn](https://scikit-learn.org/)** — the baseline linear regression model
- **[pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)** — data
  processing
- **[Plotly](https://plotly.com/python/) / [Matplotlib](https://matplotlib.org/)** —
  charts
- **[ReportLab](https://www.reportlab.com/)** — PDF report generation

---

## 10. Troubleshooting

- **"Unable to load dataset"** — make sure your CSV has at least a `date` and a
  `sales` (or equivalent) column, and that `date` values are actual dates.
- **"At least 30 daily records are recommended for forecasting"** — pick a wider date
  range or a product/store with more sales history; Prophet needs enough data points
  to learn a pattern.
- **The `prophet` package fails to install** — see the note in
  [Installation](#installation) above.
- **The demo CSV looks like a tiny text file, not real data** — you haven't pulled it
  from Git LFS yet; see [Data storage note](#7-data-storage-note-git-lfs).
