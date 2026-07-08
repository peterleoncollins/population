# population
To trial different New Zealand population and immigration strategies, assuming various fertility patterns.
<br><br>The method generates Population.csv to the model specified by the parameters fertility, dispora, returns, immigration, population size, from the Statistics NZ tabulation of latest population cohorts, and the latest cohorts' life tables, using the python program: Population.py.
<br><br>The output Population.csv is an input file to the video generator Pyramid.py which reads Population.csv and produces the video Pyramid.mp4, which illustrates the result of the parameter choices over the selected life span.
<br><br>The parameters for Population.py are:
<br>Projection years — how many years the model runs forward. Longer runs let slow effects (fertility decline, mortality fluctuation, target convergence) fully play out; too short a run may cut off trends before they matter.
<br>Base year — the calendar year the starting population applies to. Only relabels the output's year axis; doesn't change any dynamics.
<br>Current TFR — Native track's starting births-per-woman. Directly sets the initial birth rate; higher TFR means more age-0 entrants each year and slower/reversed decline.
<br>Annual % change in fertility — compounds TFR up or down every year. Negative values (the historical NZ pattern) shrink future birth cohorts progressively, hollowing out the pyramid's base over time.
<br>Immigration Y/N — master switch. If N, the population-target mechanism is disabled entirely and the population just floats on births/deaths/diaspora/returns alone.
<br>Diaspora % — % of the combined Native+Immigrant population leaving per year. Higher values drain population faster and increase how much immigration has to compensate to hit any target.
<br>Return % — % of the combined population returning per year. Offsets diaspora; higher values mean less net outflow and less immigration needed.
<br>Diaspora mean age — the age around which departures are centred (Normal distribution). Shifts which age band the population loses people from — younger mean age hollows out the working/childbearing ages more.
<br>Diaspora SD age — how spread out departures are around that mean. Larger SD spreads the loss across a wider age range instead of concentrating it.
<br>Returns mean age — the age around which returnees are centred. Determines which age band regains people — a higher mean age (e.g. 45) re-adds people past peak childbearing years, limiting the birth-rate benefit of returns.
<br>Returns SD age — spread of returnee ages around that mean. Wider spread smooths the re-entry across more age groups.
<br>Mortality fluctuation amplitude % (under 65 / 65+) — how much the death rate wobbles above/below its fixed 2025 value each cycle. Bigger amplitude means more year-to-year variation in deaths and a less "ruler-drawn" pyramid outline, with no net long-run drift.
<br>Mortality fluctuation period (years) — length of one full up-down cycle. Shorter periods produce more frequent, choppier wobbles in the pyramid; longer periods produce slower, broader waves.
<br>Fertility age-shape max shift (years, asymptotic) — the furthest the childbearing-age peak can drift later, ever. Bigger values allow more eventual delay in typical mother's age; capped so it can't run away to implausible ages.
<br>Fertility age-shape shift half-life (years) — how fast that shift approaches its cap. Shorter half-life means the peak-age delay happens mostly in the earlier decades of the run.
<br>Starting population (million) — the population-target trajectory's starting level. Sets where immigration begins trying to hold the population.
<br>Population annual change % — how much that target itself grows or shrinks each year. Positive values mean immigration will keep topping up to a rising target; negative values mean the target falls, but since immigration only ever adds (never removes) people, a falling target can only be met if natural decline is already outpacing it.
<br>Retirement age / Care age / Percent care group needing care / Carer ratio — pure reporting cutoffs (who counts as "worker," who counts as "care group," how many of those need care, and staff-per-cared ratio). These don't feed back into demographics at all — they just relabel and recompute the sociological summary rows from the same population numbers.
