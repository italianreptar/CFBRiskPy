# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:46:47 2022

@author: Connor
"""

#
# Imports
#
import requests as reqs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

from pathlib import Path
from scipy.special import erf

_BASE ="https://collegefootballrisk.com/api"
_SEASON = 3
plt.style.use("bmh")

def yline(loc, *args, ax=None, **kwargs):
    if ax is None:
        ylims = plt.ylim()
        plt.plot([loc, loc], ylims, *args, **kwargs)
        plt.ylim(ylims)
    else:
        ylims = ax.get_ylim()
        ax.plot([loc, loc], ylims, *args, **kwargs)
        ax.set_ylim(ylims)

def xline(loc, *args, ax=None, **kwargs):
    if ax is None:
        xlims = plt.xlim()
        plt.plot(xlims, [loc, loc], *args, **kwargs)
        plt.xlim(xlims)
    else:
        xlims = ax.get_xlim()
        ax.plot(xlims, [loc, loc], *args, **kwargs)
        ax.set_xlim(xlims)

def create_total_roll_log_hist(
        end_day,
        season=_SEASON,
        save_dir=None):
    """
    ``create_total_roll_log_hist`` will scrape the roll logs from day 1 to
    today to get the normalized random numbers from the roll to verify each
    day's roll conforms to a uniform distribution as a quick visual test that
    no fishyness is going on under the hood.

    Parameters
    ----------
    end_day : int
        Last day where the roll api is populated
    season : int
        Season parameter to api call
    save_dir : str, optional
        Directory to save the plot to. The default is None.

    Returns
    -------
    Data.

    """
    overall_rolls_list = list()
    total_days_array = np.array([])
    for day in range(1, end_day+1):
        roll_log_req = reqs.get(_BASE+"/roll/log",
                             params={"season": season,
                                     "day": day})
        roll_log_info = roll_log_req.json()
        terry_rolls = roll_log_info["territoryRolls"]
        overall_rolls_list.append(list())

        for terry in terry_rolls:
            if terry["randomNumber"]:
                terry_turn_req = reqs.get(_BASE+"/territory/turn",
                             params={"territory": terry["territory"],
                                     "season": season,
                                     "day": day})
                terry_turn_info = terry_turn_req.json()
                total_power = np.sum([team["power"] for team in terry_turn_info["teams"]])

                overall_rolls_list[-1].append(terry["randomNumber"] / total_power)

        total_days_array = np.append(total_days_array, overall_rolls_list[-1])
        fig = plt.figure()
        _ax = plt.gca()
        # TODO: Add Mean, std
        # Note: expected stddev is sqrt(1/12)
        # Also, add number of rolls, since that varies per day
        # I wanna know how many total are done
        heights, bin_sf = np.histogram(overall_rolls_list[-1], np.arange(0, 1.05, 0.1))
        mu, sigma = np.mean(overall_rolls_list[-1]), np.std(overall_rolls_list[-1])
        bin_size = (bin_sf[1]-bin_sf[0])
        xx = [(vv + bin_sf[ii+1])/2 for ii, vv in enumerate(bin_sf[:-1])]
        _ax.bar(xx, heights/np.sum(heights), bin_size*0.97, align="center", color="#24478f", edgecolor="#ffbb99")
        xline(0.1, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#081840")
        _ax.set_title(f"Plot Day {day} Normalized Rolls")
        _ax.set_xlabel(f"Bins of size {bin_size}")
        _ax.set_ylabel("Percent Chance to Land in Designated Bin")
        my_anno_text = f"""$\mu = {mu:2.3f}$
$\sigma = {sigma:2.3f}$"""
        _ax.text(0.03,
                 0.10,
                 my_anno_text,
                 bbox={'facecolor': 'white', 'alpha': 0.7},
                 transform=_ax.transAxes)
        if save_dir is not None:
            fig.savefig(save_dir / f"day_{day}_roll_hist.png", dpi=150)

        fig = plt.figure()
        _ax = plt.gca()
        heights, bin_sf = np.histogram(total_days_array, np.arange(0, 1.05, 0.1))
        mu, sigma = np.mean(total_days_array), np.std(total_days_array)
        bin_size = (bin_sf[1]-bin_sf[0])
        xx = [(vv + bin_sf[ii+1])/2 for ii, vv in enumerate(bin_sf[:-1])]
        _ax.bar(xx, heights/np.sum(heights), bin_size*0.97, align="center", color="#003366", edgecolor="#ffd480")
        _ax.set_title(f"Aggregate Histogram for all days combined up to Day {day}\n({len(total_days_array)} rolls, normalized)")
        xline(0.1, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#081840")
        _ax.set_xlabel(f"Bins of size {bin_size}")
        _ax.set_ylabel("Number of Normalized Rolls in a Bin Range")
        my_anno_text = f"""$\mu = {mu:2.3f}$ (${0.5-mu:2.3f}$ off expected)
$\sigma = {sigma:2.3f}$ (${np.sqrt(1/12)-sigma:2.3f}$ off expected)"""
        _ax.text(0.03,
                 0.10,
                 my_anno_text,
                 bbox={'facecolor': 'white', 'alpha': 0.7},
                 transform=_ax.transAxes)

        if save_dir is not None:
            fig.savefig(save_dir / f"day_{day}_total_roll_hist.png", dpi=150)

    return overall_rolls_list


def create_expected_value_hist(
        team_name,
        rank,
        day,
        prev_num_terry,
        num_runs=100000,
        season=_SEASON,
        axis=None,
        save_dir=None):
    """
    ``create_expected_value_hist``, as the name suggests, creates an expected
    value histogram for a given team and day from the data in the CFB_RISK api.

    if ax = None, plt.gca() is used.
    """
    try:
        team_odds_req = reqs.get(_BASE+"/team/odds",
                             params={"season": season,
                                     "day": day,
                                     "team": team_name})
        team_odds_info = team_odds_req.json()

        teams_req = reqs.get(_BASE+"/teams")
        team_info = teams_req.json()

        p_color = None
        for team in team_info:
            if team["name"] == team_name:
                p_color = team["colors"]["primary"]
                s_color = team["colors"]["secondary"]
                break

        if p_color is None:
            raise ValueError(f"Invalid team_name = {team_name}")

        p_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(p_color[5:-1].split(",")))
        s_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(s_color[5:-1].split(",")))

        if p_color[0:3] == (1, 1, 1):
            p_color = (0, 0, 0, p_color[3])
        if s_color[0:3] == (1, 1, 1):
            s_color = (0, 0, 0, s_color[3])

        num_territories = len(team_odds_info)
        # start with a vector of ones (the "empty territories have a chance of 1)
        odds = np.ones((num_territories,))

        # for each territoy, exluding "all", compute exact odds
        odds = [tory["teamPower"]/tory["territoryPower"]  # put the stats, else 1
                    if tory["territoryPower"]>0 else 1 # if denom != 0
                    for tory in team_odds_info] # for tory in odds_info

        # This calculates the PDF
        vals = 1
        for k in odds:
            vals = np.convolve(vals, [1-k, k])

        # axis handling
        if axis is None:
            fig = plt.figure()
            _ax = plt.gca()
        else:
            _ax = axis

        # set up plot values
        act = sum([1 if tory["winner"] == team_name else 0 for tory in team_odds_info])
        exp = sum(odds)
        # Gets the Expected Value numerically to validate expected Odds
        mu = np.sum(vals*np.arange(len(vals)))
        # Gets the Sigma numerically to validate variance
        sigma = np.sqrt(sum(vals*(np.arange(len(vals)) - mu)**2))
        dsigma = (act-mu) / sigma
        # draw_percentage = stats.norm.pdf(dsigma)*100

        if dsigma < 0:
            act_color = "#781b0e"
        else:
            act_color = "#3b8750"

        x = np.linspace(0, num_territories, 5000)
        y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * \
            (np.power(np.e, -(np.power((x - mu), 2) / (2 * np.power(sigma, 2)))))
        cdf = 0.5 *  (1 + erf((act-exp)/(np.sqrt(2)*sigma)))
        _ax.plot(x,y*100, linestyle="-", linewidth=0.5, color="#54585A", label="$X$ ~ $N(\mu, \sigma)$")
        _ax.bar(np.arange(num_territories+1), vals*100, 0.9, align="center", color=p_color, edgecolor=s_color)
        yline(exp, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#081840", label="Expected Value")
        yline(act, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color=act_color, label="Actual Territories")
        yline(prev_num_terry, ax=_ax, linestyle=(0,(1,1)), linewidth=2, color="#ffb521", label="Prev Num. Territories")
        dT = act - prev_num_terry
        _ax.set_title(f"Number of Territories Histogram: {team_name}\n$Expected: {exp:2.2f}$, $Actual: {act}$, $\Delta Territories = {dT}$")
        _ax.set_xlabel("Number of Territories Won")
        _ax.set_ylabel("Percent Chance to Win N Territories (%)")
        my_anno_text = f"""$\mu = {mu:2.3f}$
$3\sigma = {3*sigma:2.3f}$
$\Delta\sigma = {dsigma:2.3f}$
$P(Draw) = {100*vals[act]:2.3f}\%$"""

        x_min, x_max = _ax.get_xlim()
        y_min, y_max = _ax.get_ylim()
        if (mu) < (x_max-x_min)//2:
            # put both on right:
            _ax.legend(loc="upper right")
            _ax.text(0.72,
                     0.08,
                     my_anno_text,
                     bbox={'facecolor': 'white', 'alpha': 0.7},
                     transform=_ax.transAxes)
        elif vals[0] > 5:
            # top
            _ax.legend(loc="upper left")
            _ax.text(0.72,
                     0.80,
                     my_anno_text,
                     bbox={'facecolor': 'white', 'alpha': 0.7},
                     transform=_ax.transAxes)
        else:
            # left
            _ax.legend(loc="upper left")
            _ax.text(0.03,
                     0.10,
                     my_anno_text,
                     bbox={'facecolor': 'white', 'alpha': 0.7},
                     transform=_ax.transAxes)

        if save_dir is not None:
            if len(str(rank)) == 1:
                strank = f"0{rank}"
            else:
                strank = f"{rank}"
            fig.savefig(save_dir / f"{strank}_{team_name.lower().replace(' ', '_')}_territory_hist.png", dpi=150)

        return mu, sigma, dsigma, act, cdf
    except:
        print("someting wrong")

def create_cumulative_dsig(
        dsigs,
        teams,
        season=_SEASON,
        axis=None,
        save_dir=None):
    """
    ``create_cumulative_dsig``, as the name suggests, creates the delta sigma
    chart for a given set of teams up from the data in the CFB_RISK api.

    if ax = None, plt.gca() is used.
    if save_dir is not none, the chart will be saved.
    """
    # axis handling
    if axis is None:
        fig = plt.figure(figsize=(11, 8.5))
        _ax = plt.gca()
    else:
        _ax = axis
    s_teams = sorted(teams)
    for team_name in s_teams:
        try:
            teams_req = reqs.get(_BASE+"/teams")
            team_info = teams_req.json()

            p_color = None
            for team in team_info:
                if team["name"] == team_name:
                    p_color = team["colors"]["primary"]
                    s_color = team["colors"]["secondary"]
                    break

            if p_color is None:
                raise ValueError(f"Invalid team_name = {team_name}")

            p_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(p_color[5:-1].split(",")))
            s_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(s_color[5:-1].split(",")))

            if p_color[0:3] == (1, 1, 1):
                p_color = (0, 0, 0, p_color[3])
            if s_color[0:3] == (1, 1, 1):
                s_color = (0, 0, 0, s_color[3])

            # set up plot values
            x = np.arange(0, len(dsigs[team_name]))
            y = dsigs[team_name]
            markeredgecolor = p_color
            marker="o"
            linestyle = "-"
            if team_name == "Alabama":
                marker="s"
            elif team_name == "Chaos":
                marker="."
            elif team_name == "Georgia Tech":
                marker="h"
            elif team_name == "Iowa State":
                marker="P"
            elif team_name == "Michigan":
                marker="^"
            elif team_name == "Nebraska":
                marker="p"
            elif team_name == "Ohio State":
                marker="D"
            elif team_name == "Tennessee":
                marker="2"
            elif team_name == "Texas":
                marker="*"
            elif team_name == "Texas A&M":
                marker="*"
            elif team_name == "Wisconsin":
                marker="X"
            else:
                pass
            _ax.plot(x,y, marker=marker, markeredgecolor=markeredgecolor, linestyle=linestyle, linewidth=2.5, color=p_color, label=team_name)

        except Exception as inst:
            print("someting wrong", inst)
    _ax.set_title("$\sum_{n=1}^{day}\Delta\sigma_{n}$ Chart")
    _ax.set_xlabel("Turn Day")
    _ax.set_ylabel("Cumulative $\sum\Delta\sigma$")
    _ax.legend(loc="best")
    if save_dir is not None:
        fig.savefig(save_dir / "cumulative_delta_sigma_chart.png", dpi=150)

def create_running_avg_dsig(
        dsigs,
        teams,
        season=_SEASON,
        axis=None,
        save_dir=None):
    """
    ``create_cumulative_dsig``, as the name suggests, creates the delta sigma
    chart for a given set of teams up from the data in the CFB_RISK api.

    if ax = None, plt.gca() is used.
    if save_dir is not none, the chart will be saved.
    """
    # axis handling
    if axis is None:
        fig = plt.figure(figsize=(11, 8.5))
        _ax = plt.gca()
    else:
        _ax = axis
    s_teams = sorted(teams)
    for team_name in s_teams:
        try:
            teams_req = reqs.get(_BASE+"/teams")
            team_info = teams_req.json()

            p_color = None
            for team in team_info:
                if team["name"] == team_name:
                    p_color = team["colors"]["primary"]
                    s_color = team["colors"]["secondary"]
                    break

            if p_color is None:
                raise ValueError(f"Invalid team_name = {team_name}")

            p_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(p_color[5:-1].split(",")))
            s_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(s_color[5:-1].split(",")))

            if p_color[0:3] == (1, 1, 1):
                p_color = (0, 0, 0, p_color[3])
            if s_color[0:3] == (1, 1, 1):
                s_color = (0, 0, 0, s_color[3])

            # set up plot values
            x = np.arange(1, len(dsigs[team_name])+1)
            y = dsigs[team_name] / x
            markeredgecolor = p_color
            if team_name == "ACME":
                linestyle = ":"
            elif team_name == "Kum and Go":
                linestyle = "-"
                p_color = "#fff3a8"
                markeredgecolor = "k"
            else:
                linestyle = "-"

            _ax.plot(x,y, marker="o", markeredgecolor=markeredgecolor, linestyle=linestyle, linewidth=2.5, color=p_color, label=team_name)

        except Exception as inst:
            print("someting wrong", inst)
    _ax.set_title("$\sum_{n=1}^{day}\Delta\sigma_{n} / {n}$ (Running Avg.) Chart")
    _ax.set_xlabel("Turn/Day")
    _ax.set_ylabel("Running Avg. $\Delta\sigma$")
    _ax.legend(loc="best")
    if save_dir is not None:
        fig.savefig(save_dir / "running_avg_delta_sigma_chart.png", dpi=150)

def create_all_hists(
        day,
        season=_SEASON,
        save_dir=None
        ):
    leader_req = reqs.get(_BASE+"/stats/leaderboard",
                         params={"season": season,
                                 "day": day})
    leaders = leader_req.json()
    if day > 1:
        leader_req_yest = reqs.get(_BASE+"/stats/leaderboard",
                             params={"season": season,
                                     "day": day-1})
        leader_yest = leader_req_yest.json()

    mu = np.ones((len(leaders),))
    sig = np.ones((len(leaders),))
    dsig = np.ones((len(leaders),))
    dsig_dict = {}
    act = np.ones((len(leaders),))
    for ind, leader in enumerate(leaders):
        print("Making hist for: ", leader["name"])
        if day > 1:
            prev_num_terry = [ll for ll in leader_yest if ll["name"] == leader["name"]]
            if prev_num_terry:
                prev_num_terry = prev_num_terry[0]["territoryCount"]
            else:
                prev_num_terry = 0
        else:
            prev_num_terry = leader["territoryCount"]
        try:
            mu[ind], sig[ind], dsig[ind], act[ind], cdf = create_expected_value_hist(
                leader["name"],
                leader["rank"],
                day,
                int(prev_num_terry),
                season=season,
                save_dir=save_dir)
            dsig_dict[leader["name"]] = dsig[ind]
        except TypeError as inst:
            print("Unable to make hist for ", leader["name"], ". May not have any players today.")
            print(inst)

    return (min(dsig), leaders[np.argmin(dsig)]["name"]), (max(dsig), leaders[np.argmax(dsig)]["name"]), dsig_dict

def main(day=None):
    date = datetime.date
    # Set this true if you want to save the graphs
    SAVE_FLAG = True
    # Set this true if you want to replace the current existing graphs
    REPLACE_FLAG = True

    if SAVE_FLAG:
        output_directory = r"D:\Connor\Documents\GA 2023\PyProjects\CFBRiskPy\cfb_artifacts"
        figs_base_dir = Path(output_directory)
        check_dir = figs_base_dir / f"{date.today().isoformat()}"
        # check_dir = figs_base_dir / "2020-04-22"
        asserted_dir = figs_base_dir / "temp_dir"
        # asserted_dir = check_dir
        if not check_dir.exists():
            os.mkdir(check_dir)
            save_dir = check_dir
        else:
            if REPLACE_FLAG:
                save_dir = check_dir
            else:
                save_dir = asserted_dir
    else:
        save_dir = None

    # Get delta Time since start of game
    if not day:
        dt_now = datetime.datetime.now()
        deltaT = dt_now-datetime.datetime(2023, 1, 20)
        day = deltaT.days  # get just the delta number of days

    # print(f"Generating plots for day = {day}...")
    # mins_team, max_team, dsig_dict = create_all_hists(day, save_dir=save_dir)

    # data = create_total_roll_log_hist(
    #     day,
    #     season=_SEASON,
    #     save_dir=save_dir)

    dsig_dicts = []
    start = day
    end = day+1
    for dd in range(start, end):
        sd = save_dir if dd == day else None
        print(f"Generating plots for day = {dd}...")
        mins_team, max_team, dsig_dict = create_all_hists(dd, save_dir=sd)
        dsig_dicts.append(dsig_dict.copy())
        plt.close("all")

    # Wants:
    # Want to determine who got the overall unluckiest and luckiest
    # (CumSum dsig chart, only the leaders on the board)
    chosen_ones = dsig_dicts[-1].copy()
    cumsum_dsig_data = {}
    for team in chosen_ones.keys():
        cumsum_dsig_data[team] = np.ones((len(range(start,end)),))*np.nan
        cumsum_dsig_data[team][0] = 0

    this_start = start if start != 0 else 1
    for dd in range(0, end-this_start):
        for team in chosen_ones.keys():
            if team in dsig_dicts[dd]:
                cumsum_dsig_data[team][dd] = dsig_dicts[dd][team]

    last_roll_day = (datetime.date.today()-datetime.timedelta(days=1))
    if last_roll_day.isoweekday() == 7:
        last_roll_day =(datetime.date.today()-datetime.timedelta(days=2))
    last_roll_iso_date = last_roll_day.isoformat()

    if Path(f"cfb_artifacts/cumsum_dsig_data_{last_roll_iso_date}.csv").exists():
        # do load history data
        df = pd.read_csv(f"cfb_artifacts/cumsum_dsig_data_{last_roll_iso_date}.csv")
        # append todays new data
        df_today = pd.DataFrame(cumsum_dsig_data)
        new_df = df.append(df_today)
        # write to file
        new_df.to_csv(f"cfb_artifacts/cumsum_dsig_data_{datetime.date.today().isoformat()}.csv", index=False)
    else:
        # don't overwrite history data, but treat it like it was run from day 0
        new_df = pd.DataFrame(cumsum_dsig_data)
        new_df.to_csv(f"cfb_artifacts/cumsum_dsig_data_{datetime.date.today().isoformat()}.csv", index=False)
    cumsum_dsig_data = new_df.to_dict("list")

    cumsum_dsigs = {}
    for team in chosen_ones.keys():
        cumsum_dsigs[team] = np.nancumsum(cumsum_dsig_data[team])

    create_cumulative_dsig(
        cumsum_dsigs,
        chosen_ones.keys(),
        save_dir=save_dir
    )

    # create_running_avg_dsig(
    #     cumsum_dsigs,
    #     chosen_ones.keys(),
    #     save_dir=save_dir
    # )

    return cumsum_dsigs, cumsum_dsig_data, dsig_dicts

def run_basic_monte(day, num_runs=100000, save_dir=None):
    maxes = np.zeros((num_runs,))
    mins = np.zeros((num_runs,))
    means = np.zeros((num_runs,))
    sums = np.zeros((num_runs,))

    minmax = 3
    maxmax = 0

    minmin = 0
    maxmin = -3

    minsum = 0
    maxsum = 0

    minmean = 0
    maxmean = 0
    for i in range(num_runs):
        run = np.random.randn(day,)
        maxes[i], mins[i], means[i], sums[i] = np.max(run), np.min(run), np.mean(run), np.sum(run)

        if maxes[i] > maxmax:
            maxmax = maxes[i]
            maxmaxrun = run
        if maxes[i] < minmax:
            minmax = maxes[i]
            minmaxrun = run
        if mins[i] > maxmin:
            maxmin = mins[i]
            maxminrun = run
        if mins[i] < minmin:
            minmin = mins[i]
            minminrun = run
        if sums[i] > maxsum:
            maxsum = sums[i]
            maxsumrun = run
        if sums[i] < minsum:
            minsum = sums[i]
            minsumrun = run

    # TODO: Make a hist for each
    # TODO: Make the max sum and min sum bounds

    fig = plt.figure(figsize=(11,8.5))
    _ax = plt.gca()
    heights, bin_sf = np.histogram(sums, np.linspace(minsum, maxsum, 25))
    mu, sigma = np.mean(sums), np.std(sums)
    bin_size = (bin_sf[1]-bin_sf[0])
    xx = [(vv + bin_sf[ii+1])/2 for ii, vv in enumerate(bin_sf[:-1])]
    _ax.bar(xx, 100*heights/np.sum(heights), bin_size*0.97, align="center", color="#003366", edgecolor="#ffd480")
    _ax.set_title("Histogram of Monte Results for $\sum_{n=1}^{day}\Delta\sigma_{n}$")
    yline(0.0, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#e3c21e")
    _ax.set_xlabel(f"Bins of size {bin_size:2.3f}")
    _ax.set_ylabel("Percent of Sums in a Bin")
    my_anno_text = f"""$\mu = {mu:2.3f}$
$\sigma = {sigma:2.3f}$
$min(sums) = {minsum:2.3f}$
$max(sums) = {maxsum:2.3f}$"""
    _ax.text(0.03,
             0.775,
             my_anno_text,
             bbox={'facecolor': 'white', 'alpha': 0.7},
             transform=_ax.transAxes)

    if save_dir is not None:
        fig.savefig(save_dir / f"monte_sum_sig_hist.png", dpi=150)
    return (maxmax, maxmaxrun), (minmax, minmaxrun), (maxmin, maxminrun), (minmin, minminrun), (minsum, minsumrun), (maxsum, maxsumrun)


if __name__ == "__main__":
    # 3/3
    day = 36

    # minteam, maxteam, data, dsig_dict = main(day)
    # print(minteam, maxteam)
    cumsum_dsigs, cumsum_dsig_data, dsig_dicts = main(day)
    values = np.array(list(dsig_dicts[-1].values()))
    keys = np.array(list(dsig_dicts[-1].keys()))
    inds = np.argsort(values)

    print("\nLuckiest Teams | Today's Sigma\n------------- | --------------")
    for ii in range(-1, -4, -1):
        print(keys[inds[ii]], values[inds[ii]], sep=" | ")

    print("\nUnluckiest Teams | Today's Sigma\n------------- | --------------")
    for ii in range(0,3):
        print(keys[inds[ii]], values[inds[ii]], sep=" | ")


    #%%
    num_teams = len(keys)
    data = np.random.randn(num_teams,10000)
    maxdata = np.max(data, 0)
    mindata = np.min(data, 0)
    fig = plt.figure(figsize=(11,8.5))
    _ax = plt.gca()
    minheights, minbin_sf = np.histogram(mindata, np.linspace(min(mindata), max(mindata), 25), density=True)
    minmu, minsigma = np.mean(mindata), np.std(mindata)
    minbin_size = (minbin_sf[1]-minbin_sf[0])
    xx = [(vv + minbin_sf[ii+1])/2 for ii, vv in enumerate(minbin_sf[:-1])]
    _ax.bar(xx, 100*minheights/np.sum(minheights), minbin_size*0.97, align="center", color="#003366", edgecolor="#ffd480")
    # _ax.set_title("Histogram of Mins and Maxes of 15 $\sigma$ draw values")
    yline(minmu, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#000000")
    yline(minmu+minsigma, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#e3c21e")
    yline(minmu-minsigma, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#e3c21e")

    maxheights, maxbin_sf = np.histogram(maxdata, np.linspace(min(maxdata), max(maxdata), 25), density=True)
    maxmu, maxsigma = np.mean(maxdata), np.std(maxdata)
    maxbin_size = (maxbin_sf[1]-maxbin_sf[0])
    xx = [(vv + maxbin_sf[ii+1])/2 for ii, vv in enumerate(maxbin_sf[:-1])]
    _ax.bar(xx, 100*maxheights/np.sum(maxheights), maxbin_size*0.97, align="center", color="#003366", edgecolor="#ffd480")
    _ax.set_title(f"Histogram of Mins and Maxes of {num_teams} $\sigma$ draw values\n$\mu: \pm{maxmu:1.3f}; \sigma: {maxsigma:1.3f}$")
    yline(maxmu, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#000000")
    yline(maxmu+maxsigma, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#e3c21e")
    yline(maxmu-maxsigma, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#e3c21e")
    _ax.set_ylabel("Percent of values within any bin")
