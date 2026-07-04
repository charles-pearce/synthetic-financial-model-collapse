import pandas as pd
from src.processes import Merton
from src.processes import GBM
from src.recursion import run_recursion
from src.processes import Heston
from src.quantgan import train_quantgan

# Each block below runs one experimental configuration and writes its metric
# records to a CSV in the results directory. The active block is the one not
# commented out; the commented blocks record the configurations used for the
# other experiments in the thesis and can be reactivated to reproduce them.

# Replacement
all_records = []
for seed in range(15):
    print(f"\n=== seed {seed} ===")
    records = run_recursion(
        process=Heston(),
        condition="replacement",
        alpha=1.0,
        n_generations=15,
        n_iters=3000,
        seed=seed,
    )
    all_records.extend(records)

df = pd.DataFrame(all_records)
df.to_csv("results/wgangp_heston_replacement2.csv", index=False)
print(f"\nSaved {len(df)} rows to results/wgangp_heston_replacement2.csv")

# Accumulation for WGAN-GP
# all_records = []
# alphas = [0.9, 0.75, 0.5]
# for seed in range(10):
#     print(f"\n=== seed {seed} ===")
#     for a in alphas:
#         records = run_recursion(
#             process=Merton(),
#             condition="accumulation",
#             alpha=a,
#             n_generations=15,
#             n_iters=3000,
#             seed=seed,
#         )
#         all_records.extend(records)
#     pd.DataFrame(all_records).to_csv("results/wgangp_merton_accumulation.csv", index=False)
#     print(f"  saved {len(all_records)} rows so far")


# Replacement for QuantGAN, seeds ten to fourteen
# all_records = []
# for seed in range(10, 15):
#     print(f"\n=== seed {seed} ===")
#     records = run_recursion(
#         process=Heston(),
#         condition="replacement",
#         alpha=1.0,
#         n_generations=15,
#         n_iters=1500,
#         seed=seed,
#         train_fn=train_quantgan,
#         noise_shape=(5000, 3, 100),
#     )
#     all_records.extend(records)
#     pd.DataFrame(all_records).to_csv("results/quantgan_heston_replacement_full10.csv", index=False)
#     print(f"  saved {len(all_records)} rows")

# Accumulation for QuantGAN on the Heston process
# all_records = []
# alphas = [0.9, 0.75, 0.5]
# for seed in range(8, 10):
#     print(f"\n=== seed {seed} ===")
#     for a in alphas:
#         records = run_recursion(
#             process=Heston(),
#             condition="accumulation",
#             alpha=a,
#             n_generations=15,
#             n_iters=1500,
#             seed=seed,
#             train_fn=train_quantgan,
#             noise_shape=(5000, 3, 100),
#         )
#         all_records.extend(records)
#     pd.DataFrame(all_records).to_csv("results/quantgan_heston_accumulation8-9.csv", index=False)
#     print(f"  saved {len(all_records)} rows so far")
