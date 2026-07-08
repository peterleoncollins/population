"""
NZ population projection - cohort component, single-year age bands, female-only
(doubled to total). Two tracks per age: Native, Immigrant.

INPUTS -- real Stats NZ Infoshare exports, used as-is:
1. POP_PATH: "Estimated Resident Population by Age and Sex (1991+) (Annual-Dec)",
   table DPE403905, female, single year of age, December 2025.
2. MORT_PATH: "Age-specific death rates by sex, December years (total
   population)", table DMM001AA, female, 5-year age bands, per 1,000, 2025.
   Converted to px = 1 - rate/1000, held constant. Same px used for both
   tracks (no differential immigrant/native mortality assumed).
https://www.stats.govt.nz/ (Infoshare, both tables, downloaded by user)

FERTILITY SHAPE: Stats NZ, "Births and deaths: Year ended December 2024",
age-specific birth rates per 1,000 women, 5-year bands.
https://www.stats.govt.nz/information-releases/births-and-deaths-year-ended-december-2024/
Native track TFR follows the entered current-TFR / annual-%-change path.
The age-SHAPE shifts later over time, but ASYMPTOTICALLY -- shift(t) =
max_shift x (1 - 0.5^(t/half_life)), converging toward a bounded maximum
(runtime parameter, e.g. 4 years) rather than drifting indefinitely. A
linear years-per-decade shift (an earlier version of this model) has no
limit and eventually pushes the fertility peak to biologically implausible
ages over a century; the reproductive age window itself is bounded, so the
shift in mean childbearing age must be bounded too. Immigrant track TFR
held constant at 1.7 (user-specified), unshifted 2024 shape.

MORTALITY: Stats NZ, "Age-specific death rates by sex, December years",
table DMM001AA, 2025, converted to px = 1 - rate/1000. FLUCTUATES around the
fixed 2025 rate rather than drifting in one direction: qx(age,t) =
qx_2025(age) x (1 + amplitude(age) x sin(2*pi*t/period)) -- a symmetric
sinusoidal wobble, two-tier amplitude (under 65 / 65+, runtime parameters),
shared period (runtime parameter, years per cycle). An earlier version of
this model compounded a constant %/year improvement forever, which drove qx
toward zero and stalled the 95+ open-ended age group entirely; a later
version converged toward a floor instead, which still drifted in one
direction. This version does neither -- it wobbles around the 2025 baseline
with no net drift, per your specification. Deterministic and reproducible
(a fixed sinusoid, not a random-number draw); amplitude and period are
user-specified, not sourced.

SEX RATIO AT BIRTH: 105.5 males per 100 females, Stats NZ methodology.
https://datainfoplus.stats.govt.nz/Item/nz.govt.stats/583ca9da-d6d2-41e0-b626-5743c14deaf5/128

MIGRATION MECHANISM (this run's addition, not a Stats NZ input):
Diaspora/Returns: each year, Diaspora % and Return % (entered at runtime, one
decimal place) are applied to the PRECEDING year's COMBINED Native+Immigrant
Total (start-of-year, before this year's flows) to get whole-number persons
leaving/returning -- using the preceding total avoids a circular dependency
on the year's own still-unknown total. Combined basis because many
immigrants use NZ as a stepping stone onward (e.g. to Australia).
Departures are distributed across ages by a NORMAL shape, unrestricted/not
truncated (mean age and SD as runtime parameters, e.g. mean=26, SD=3) --
user-specified, not sourced. (Real NZ-citizen departures are in fact
right-skewed young rather than symmetric -- 38% aged 18-30 in 2025: Stats
NZ, "Net migration gain of 14,200",
https://www.stats.govt.nz/news/net-migration-gain-of-14200/ -- but you've
asked for an unrestricted normal specifically, so no skew is imposed.)
Returns are distributed by a NORMAL shape, unrestricted/not truncated (mean
age and SD as runtime parameters, e.g. mean=45, SD=10) -- this describes the
AGE OF RETURNEES directly, not a lag applied to specific departure cohorts
(no departure-cohort ledger is kept). User-specified, not sourced. The net per-age amount is then split
between the Native and Immigrant tracks in proportion to each track's own
share of that age's population, after deaths are counted (not counted as
deaths). Births still split Native/Immigrant at age 0 for audit, then merge
into the Native track's future ageing as before.
Immigration: target-based, minimum zero (no forced-emigration lever).
Target_t = Starting population x (1 + Population annual change %)^t.
Population_t = MAX(Target_t, Natural_t), where Natural_t is what births,
deaths, and diaspora/returns alone would produce. The target only binds
when Immigration=Y AND the target is above Natural_t; otherwise immigration
is zero and population floats freely (a falling target cannot force
population down, since there is no negative-immigration/forced-emigration
mechanism). Immigrants spread evenly across single years of age 18-35
inclusive. All births, including births to immigrant mothers, are credited
to the Native track next year (the 1.7 TFR applies only to the first
generation of immigrant women, not their NZ-born daughters) -- stated
assumption, not sourced.
"""

import numpy as np
import pandas as pd

MAX_AGE = 95
POP_PATH = "DPE403905_20260702_044540_72.csv"
MORT_PATH = "DMM168901_20260702_043705_39.csv"
SRB = 1.055
FEMALE_BIRTH_SHARE = 1 / (1 + SRB)
IMMIGRANT_TFR = 1.7
IMMIGRANT_AGE_LO, IMMIGRANT_AGE_HI = 18, 35  # inclusive

ASFR_BANDS_2024 = {
    (15, 5): 10.3, (20, 5): 44.8, (25, 5): 81.4,
    (30, 5): 102.3, (35, 5): 60.5, (40, 5): 13.5,
}


def ask(prompt, default, cast=float):
    raw = input(f"{prompt} [{default}]: ").strip()
    return cast(raw) if raw else cast(default)


def load_base_population(path):
    header = pd.read_csv(path, skiprows=3, nrows=1, header=None).iloc[0]
    data = pd.read_csv(path, skiprows=4, nrows=1, header=None).iloc[0]
    ages, values = [], []
    for col in range(1, len(header)):
        label = str(header[col])
        age = 95 if "95" in label else int(label.split()[0])
        ages.append(age)
        values.append(float(data[col]))
    pop = pd.Series(values, index=ages).groupby(level=0).sum()
    return pop.reindex(range(MAX_AGE + 1)).to_numpy(dtype=float)


def load_px_from_death_rates(path):
    header = pd.read_csv(path, skiprows=2, nrows=1, header=None).iloc[0]
    data = pd.read_csv(path, skiprows=3, nrows=1, header=None).iloc[0]
    qx_annual = np.zeros(MAX_AGE + 1)
    for col in range(1, len(header)):
        label = str(header[col])
        qx = float(data[col]) / 1000.0
        if "Over" in label:
            start, width = 90, MAX_AGE - 90 + 1
        else:
            start, width = int(label.split("-")[0]), 5
        for a in range(start, min(start + width, MAX_AGE + 1)):
            qx_annual[a] = qx
    return 1 - qx_annual


def asfr_shape_2024():
    shape = np.zeros(MAX_AGE + 1)
    for (start, width), rate in ASFR_BANDS_2024.items():
        for a in range(start, start + width):
            shape[a] = rate / 1000.0
    return shape


def diaspora_shape(mean_age, sd_age, max_age):
    """Normal age-shape for departures, unrestricted (not truncated) --
    user-specified as Normal(mean, sd), e.g. mean=26, sd=3. Unlike the
    lognormal this replaces, sigma here IS the standard deviation directly.
    At mean=26/sd=3 the mass below age 0 is negligible (~8.7 SDs away), so
    no truncation is applied, per your specification."""
    ages = np.arange(max_age + 1)
    shape = np.exp(-0.5 * ((ages - mean_age) / sd_age) ** 2)
    return shape / shape.sum()


def returns_shape(mean_age, sd_age, max_age):
    """Normal age-shape for returns, unrestricted (not truncated) --
    describes the age of returnees directly, not a lag applied to specific
    departure cohorts. User-specified, not sourced."""
    ages = np.arange(max_age + 1)
    shape = np.exp(-0.5 * ((ages - mean_age) / sd_age) ** 2)
    return shape / shape.sum()


def mortality_px_at_year(px_base, years_elapsed, fluct_pct_under65, fluct_pct_65plus,
                          period_years, max_age):
    """Survival probability FLUCTUATES around the fixed 2025 rate, rather than
    drifting toward a floor in one direction. qx(age,t) = qx_2025(age) x
    (1 + amplitude(age) x sin(2*pi*t/period)) -- a symmetric wobble around
    the baseline, two-tier amplitude (under 65 / 65+, runtime parameters),
    shared period (years for one full cycle). Deterministic and reproducible
    (a fixed sinusoid, not a random-number draw) -- true per-year randomness
    would make results non-reproducible run to run without adding
    argumentative value, and at national scale individual-level randomness
    washes out anyway (Law of Large Numbers). User-specified amplitude/
    period, not independently sourced."""
    ages = np.arange(max_age + 1)
    amplitude = np.where(ages >= 65, fluct_pct_65plus / 100, fluct_pct_under65 / 100)
    qx_base = 1 - px_base
    wobble = 1 + amplitude * np.sin(2 * np.pi * years_elapsed / period_years)
    qx_t = qx_base * wobble
    return 1 - qx_t


def shift_age_shape(shape, shift_years, max_age):
    """Shift an age-shape vector later by shift_years (fractional), via linear
    interpolation, then renormalise. Models the mean age of childbearing
    moving later over time, rather than holding the 2024 ASFR shape fixed
    for a century."""
    ages = np.arange(max_age + 1)
    shifted = np.interp(ages - shift_years, ages, shape, left=0, right=0)
    total = shifted.sum()
    return shifted / total if total > 0 else shifted


def age_forward(pop, px):
    new = np.zeros(MAX_AGE + 1)
    new[0] = 0.0  # births added by caller
    new[1:MAX_AGE] = pop[0:MAX_AGE - 1] * px[0:MAX_AGE - 1]
    new[MAX_AGE] = pop[MAX_AGE - 1] * px[MAX_AGE - 1] + pop[MAX_AGE] * px[MAX_AGE]
    return new


def main():
    print("--- Parameters (Enter to accept default) ---")
    Y = ask("Projection years", 100, int)
    base_year = ask("Base year", 2025, int)
    current_tfr = ask("Current TFR (births per woman)", 1.53, float)
    pct_change = ask("Annual % change in fertility (negative = decline)", 0.0, float)
    out_path = input("Output CSV path [Population.csv]: ").strip() \
        or "Population.csv"
    immigration_on = ask("Immigration Y/N", "Y", str).strip().upper().startswith("Y")
    diaspora_pct = round(ask("Diaspora %", 1.2, float), 1)
    return_pct = round(ask("Return %", 0.4, float), 1)
    diaspora_mean_age = ask("Diaspora mean age", 30, float)
    diaspora_sd_age = ask("Diaspora SD age", 6, float)
    returns_mean_age = ask("Returns mean age", 45, float)
    returns_sd_age = ask("Returns SD age", 10, float)
    mort_fluct_pct_under65 = ask("Mortality fluctuation amplitude % (under 65)", 0, float)
    mort_fluct_pct_65plus = ask("Mortality fluctuation amplitude % (65+)", 0, float)
    mort_period_years = ask("Mortality fluctuation period (years)", 1, float)
    fertility_max_shift_years = ask("Fertility age-shape max shift (years, asymptotic)", 0, float)
    fertility_shift_half_life = ask("Fertility age-shape shift half-life (years)", 1, float)
    start_pop_million = ask("Starting population (million)", 5.4, float)
    pop_annual_change_pct = ask("Population annual change %", 0.0, float)
    retirement_age = ask("Retirement age", 75, int)
    care_age = ask("Care age", 85, int)
    pct_needing_care = ask("Percent care group needing care", 60, float)
    carer_ratio = ask("Carer ratio", 1.3, float)

    native = load_base_population(POP_PATH)
    immigrant = np.zeros(MAX_AGE + 1)
    px_annual_base = load_px_from_death_rates(MORT_PATH)

    shape = asfr_shape_2024()
    shape_tfr = shape.sum()
    native_asfr_base = shape * (current_tfr / shape_tfr)
    immig_asfr = shape * (IMMIGRANT_TFR / shape_tfr)

    diaspora_age_shape = diaspora_shape(diaspora_mean_age, diaspora_sd_age, MAX_AGE)
    returns_age_shape = returns_shape(returns_mean_age, returns_sd_age, MAX_AGE)

    native_hist = np.zeros((MAX_AGE + 1, Y + 1))
    immig_hist = np.zeros((MAX_AGE + 1, Y + 1))
    native_hist[:, 0] = native
    immig_hist[:, 0] = immigrant

    tfr_by_year = np.zeros(Y + 1); tfr_by_year[0] = current_tfr
    deaths_female = np.zeros(Y)
    births_female_arr = np.zeros(Y)
    immigrants_total_arr = np.zeros(Y)  # doubled, units, always >= 0
    diaspora_arr = np.zeros(Y)   # total persons, whole numbers
    returns_arr = np.zeros(Y)    # total persons, whole numbers
    births_immig_age0 = np.zeros(Y + 1)  # display only, audit split of age-0 row

    n_immig_ages = IMMIGRANT_AGE_HI - IMMIGRANT_AGE_LO + 1

    for t in range(Y):
        growth_factor = (1 + pct_change / 100) ** t
        shift_years = fertility_max_shift_years * (1 - 0.5 ** (t / fertility_shift_half_life))
        shifted_native_shape = shift_age_shape(shape, shift_years, MAX_AGE)
        native_asfr_t = shifted_native_shape * current_tfr * growth_factor
        tfr_by_year[t + 1] = native_asfr_t.sum()

        px_t = mortality_px_at_year(px_annual_base, t, mort_fluct_pct_under65,
                                     mort_fluct_pct_65plus, mort_period_years, MAX_AGE)

        births_native = np.sum(native * native_asfr_t) * FEMALE_BIRTH_SHARE
        births_immig = np.sum(immigrant * immig_asfr) * FEMALE_BIRTH_SHARE
        births_female = births_native + births_immig
        births_female_arr[t] = births_female

        new_native = age_forward(native, px_t)
        new_native[0] = births_female * px_t[0]  # all newborns join Native track
        births_immig_age0[t + 1] = births_immig * px_t[0]  # audit split, display only
        new_immig = age_forward(immigrant, px_t)  # no injection yet

        deaths_female[t] = (native.sum() + immigrant.sum() + births_female) - (new_native.sum() + new_immig.sum())

        # Diaspora/Returns applied AFTER deaths are counted -- this is emigration/
        # return migration, not mortality, and must not be counted in Deaths.
        # Based on the PRECEDING Total (start-of-year, before this year's flows)
        # to avoid a circular/infinite-loop dependency on this year's own total.
        # Applied to the COMBINED Native+Immigrant population, distributed by
        # AGE-SHAPE (lognormal for departures, truncated-normal for returns --
        # see diaspora_shape/returns_shape docstrings), not flat across ages.
        # The per-age net amount is then split between Native and Immigrant in
        # proportion to each track's own share of that age's population.
        preceding_total = (native.sum() + immigrant.sum()) * 2
        diaspora_persons = round(preceding_total * diaspora_pct / 100)
        returns_persons = round(preceding_total * return_pct / 100)
        diaspora_arr[t] = diaspora_persons
        returns_arr[t] = returns_persons
        diaspora_by_age_female = diaspora_age_shape * diaspora_persons / 2
        returns_by_age_female = returns_age_shape * returns_persons / 2
        net_by_age_female = returns_by_age_female - diaspora_by_age_female

        combined_by_age = new_native + new_immig
        with np.errstate(invalid="ignore", divide="ignore"):
            native_share = np.where(combined_by_age > 0, new_native / combined_by_age, 0.5)
        new_native = new_native + net_by_age_female * native_share
        new_immig = new_immig + net_by_age_female * (1 - native_share)
        new_native = np.maximum(new_native, 0.0)
        new_immig = np.maximum(new_immig, 0.0)

        # Immigration: target-based, minimum zero (no forced emigration lever).
        # Target_t = starting population x (1 + annual %change)^t. Population_t =
        # MAX(target_t, natural_t) -- the target only binds when Immigration=Y
        # AND the target sits above what natural dynamics alone would produce;
        # below that, immigration is zero and population floats freely above target.
        natural_total_female = new_native.sum() + new_immig.sum()
        target_total = start_pop_million * 1_000_000 * (1 + pop_annual_change_pct / 100) ** t
        target_female = target_total / 2
        if immigration_on:
            population_female = max(target_female, natural_total_female)
        else:
            population_female = natural_total_female
        shortfall_female = population_female - natural_total_female  # always >= 0
        immigrants_total_arr[t] = shortfall_female * 2  # doubled, for display, never negative
        per_age = shortfall_female / n_immig_ages
        new_immig[IMMIGRANT_AGE_LO:IMMIGRANT_AGE_HI + 1] += per_age

        native, immigrant = new_native, new_immig
        native_hist[:, t + 1] = native
        immig_hist[:, t + 1] = immigrant

    years = [base_year + t for t in range(Y + 1)]
    native_units = np.round(native_hist).astype(int)
    immig_units = np.round(immig_hist).astype(int)

    header_text = (f"Projection")
    with open(out_path, "w") as f:
        f.write('"' + header_text.replace('"', '""') + '"\n')
        f.write("age,cohort," + ",".join(str(y) for y in years) + "\n")

        imm0_units = np.round(births_immig_age0).astype(int)
        native0_units = native_units[0] - imm0_units  # keeps row sum == combined total
        for a in range(MAX_AGE + 1):
            if a == 0:
                f.write("0,Native," + ",".join(f'"{v*2:,}"' for v in native0_units) + "\n")
                f.write("0,Immigrant," + ",".join(f'"{v*2:,}"' for v in imm0_units) + "\n")
            else:
                f.write(f"{a},Native," + ",".join(f'"{v*2:,}"' for v in native_units[a]) + "\n")
                f.write(f"{a},Immigrant," + ",".join(f'"{v*2:,}"' for v in immig_units[a]) + "\n")

        total_units = (native_units.sum(axis=0) + immig_units.sum(axis=0)) * 2
        f.write("Total,," + ",".join(f'"{v:,}"' for v in total_units) + "\n")
        losses = [f'"{total_units[i] - total_units[i + 1]:,}"' for i in range(len(total_units) - 1)] + [""]
        f.write("Loss,," + ",".join(losses) + "\n")

        deaths_units = np.round(deaths_female * 2).astype(int)
        f.write("Deaths,," + ",".join(f'"{v:,}"' for v in deaths_units) + ",\n")
        births_units = np.round(births_female_arr * 2).astype(int)
        f.write("Births,," + ",".join(f'"{v:,}"' for v in births_units) + ",\n")
        bd = births_units - deaths_units
        f.write("B-D,," + ",".join(f'"{v:,}"' for v in bd) + ",\n")
        immig_row = np.round(immigrants_total_arr).astype(int)
        f.write("Immigrants,," + ",".join(f'"{v:,}"' for v in immig_row) + ",\n")
        f.write("Diaspora,," + ",".join(f'"{int(v):,}"' for v in diaspora_arr) + ",\n")
        f.write("Returns,," + ",".join(f'"{int(v):,}"' for v in returns_arr) + ",\n")
        f.write("Fertility (TFR),," + ",".join(f"{v:.3f}" for v in tfr_by_year) + "\n")

        combined = native_units * 2 + immig_units * 2  # total persons by age, all years
        workers = combined[18:retirement_age].sum(axis=0)
        retired = combined[retirement_age:96].sum(axis=0)
        care_group = combined[care_age:96].sum(axis=0)
        number_cared = np.round(care_group * pct_needing_care / 100).astype(int)
        carer_staff = np.round(number_cared * carer_ratio).astype(int)
        non_care_workers = workers - carer_staff

        f.write(f"Workers <{retirement_age},," + ",".join(f'"{v:,}"' for v in workers) + "\n")
        f.write(f"Retired >={retirement_age},," + ",".join(f'"{v:,}"' for v in retired) + "\n")
        f.write(f"care group >={care_age},," + ",".join(f'"{v:,}"' for v in care_group) + "\n")
        f.write(f"Number cared {pct_needing_care:g}%,," + ",".join(f'"{v:,}"' for v in number_cared) + "\n")
        f.write(f"Carers {carer_ratio:g}:1,," + ",".join(f'"{v:,}"' for v in carer_staff) + "\n")
        f.write("non-care workers,," + ",".join(f'"{v:,}"' for v in non_care_workers) + "\n")

    print(f"\nWrote {out_path}: {2*(MAX_AGE+1)} age rows (Native+Immigrant) x {len(years)} year columns")
    print(f"Total pop {years[0]}: {total_units[0]:,}  ->  {years[-1]}: {total_units[-1]:,}")


if __name__ == "__main__":
    main()
