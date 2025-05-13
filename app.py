from alhazen import IteratedExperiment
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from pyibl import Agent, df_plot
from random import random
from shiny import reactive
from shiny.express import input, render, ui

# TODO:
#   figure out how to hide some of the plots (?)
#   maybe allow fiddling blending temperature?
#   allow dumping data as CSV
#   allow downloading plots?
#   allow saving and restoring settings?
#   online doc? probably not necessary but think about it

DEFAULT_PREPOPULATED_MULTIPLIER = 1.2


class ChoiceExperiment (IteratedExperiment):

    def prepare_experiment(self, **kwargs):
        self.noise = kwargs["noise"]
        self.decay = kwargs["decay"]
        self.temperature = kwargs["temperature"]
        self.gamble = kwargs["gamble"]
        self.prepop = kwargs["prepop"]

    def setup(self):
        self.agent = Agent(noise=self.noise, decay=self.decay, temperature=self.temperature)
        self.agent.populate(self.gamble.keys(), self.prepop)

    def run_participant_prepare(self, participant, condition, context):
        self.agent.reset(True)
        self.agent.aggregate_details = True

    def run_participant_run(self, round, participant, condition, context):
        choice = self.agent.choose(self.gamble.keys())
        u1, p2, u2 = self.gamble[choice]
        self.agent.respond(u2 if random() < p2 else u1)

    def run_participant_finish(self, participant, condition, results):
        df = self.agent.aggregate_details
        df.loc[:, "iteration"] = participant
        return df


def max_utility(g):
    return max([max(u1, u2) for u1, p2, u2 in g.values()])

@reactive.calc
def gamble():
    result = {"A": (input.A_low(), input.A_prob() / 100, input.A_high()),
              "B": (input.B_low(), input.B_prob() / 100, input.B_high())}
    if int(input.option_count()[0]) > 2:
        result["C"] = (input.C_low(), input.C_prob() / 100, input.C_high())
    if int(input.option_count()[0]) > 3:
        result["D"] = (input.D_low(), input.D_prob() / 100, input.D_high())
    return result

@reactive.calc
def prepopulated_value():
    return input.prepop_multiplier() * max_utility(gamble())

@reactive.calc
def blending_temperature():
    if input.manual_temp():
        return input.temperature()
    elif input.noise() < 0.01:
        return 1
    else:
        return math.sqrt(2) * input.noise()

@reactive.calc
def simulation_results():
    input.recompute()
    g = gamble()
    return (pd.concat(ChoiceExperiment(rounds=input.rounds(),
                                       participants=input.participants(),
                                       # On a 64 core, HyperThreaded  machine like Janus
                                       # the following translates to 56 physical cores.
                                       process_count=int(os.environ.get("WEB_PYIBL_PROCESS_COUNT", 0.43)),
                                       show_progress=False).run(gamble=g,
                                                                prepop=prepopulated_value(),
                                                                noise=input.noise(),
                                                                decay=input.decay(),
                                                                temperature=blending_temperature())))

with ui.sidebar(width=400):
    ui.tags.style("html {font-size: 50%;}")
    ui.input_select("option_count", None, ("2 Options", "3 Options", "4 Options"), width="25%")
    with ui.card():
        ui.HTML("Option A")
        with ui.layout_columns(col_widths=(3, 6, 3)):
            ui.input_numeric("A_low", None, 3)
            ui.input_slider("A_prob", None, 0, 100, 100, post="%")
            ui.input_numeric("A_high", None, 3)
    with ui.card():
        ui.HTML("Option B")
        with ui.layout_columns(col_widths=(3, 6, 3)):
            ui.input_numeric("B_low", None, 0)
            ui.input_slider("B_prob", None, 0, 100, 75, post="%")
            ui.input_numeric("B_high", None, 4)
    with ui.panel_conditional("input.option_count === '3 Options' || input.option_count === '4 Options'"):
        with ui.card():
            ui.HTML("Option C")
            with ui.layout_columns(col_widths=(3, 6, 3)):
                ui.input_numeric("C_low", None, 0)
                ui.input_slider("C_prob", None, 0, 100, 50, post="%")
                ui.input_numeric("C_high", None, 0)
    with ui.panel_conditional("input.option_count === '4 Options'"):
        with ui.card():
            ui.HTML("Option D")
            with ui.layout_columns(col_widths=(3, 6, 3)):
                ui.input_numeric("D_low", None, 0)
                ui.input_slider("D_prob", None, 0, 100, 50, post="%")
                ui.input_numeric("D_high", None, 0)
    ui.input_slider("participants", "Participants", min=1, max=2000, value=200)
    ui.input_slider("rounds", "Rounds", min=5, max=200, value=60)
    ui.input_slider("noise", "Noise", min=0, max=1.5, value=0.25)
    ui.input_slider("decay", "Decay", min=0.0, max=2.5, value=0.5)
    with ui.card():
        with ui.layout_columns():
            ui.HTML("Blending temperature")
            @render.text
            def temp_display():
                if input.manual_temp():
                    return blending_temperature()
                elif input.noise() < 0.01:
                    return "no noise ⇒ 1"
                else:
                    return f"√2 × noise = {blending_temperature():.2f}"
        ui.input_checkbox("manual_temp", "Set manually", False)
        with ui.panel_conditional("input.manual_temp"):
            ui.input_slider("temperature", None, min=0.01, max= 2.2, value=1.0)
    with ui.card():
        ui.input_switch("show_bvs", "Show blended values", True)
        ui.input_switch("show_activations", "Show activations", True)
        ui.input_switch("show_probs", "Show probabilities of retrieval", True)
    with ui.card():
        with ui.layout_columns():
            @render.text
            def prepop_value():
                return f"Prepopulated value: {prepopulated_value():.2f}"
            ui.input_switch("show_prepop", "Show in plots", False)
        with ui.layout_columns():
            @render.text
            def max_payoff():
                return f"Max payoff: {max_utility(gamble())}"
            with ui.layout_columns():
                ui.HTML("Multiplier:")
                ui.input_numeric("prepop_multiplier", None, DEFAULT_PREPOPULATED_MULTIPLIER, min=1.0)
    with ui.layout_columns(col_widths=(5, 6, 1)):
        ui.input_action_button("recompute", "Recompute")
        ui.HTML("&nbsp;")
        ui.input_dark_mode(mode="light")

def plot_thing(kind, df, max=None):
    df_plot(df, kind, xlabel="Round", show=False, max=max)

@render.plot
def plot_choice():
    plot_thing("choice", simulation_results())

with ui.panel_conditional("input.show_bvs"):
    @render.plot
    def plot_blended_values():
        plot_thing("bv", simulation_results())

with ui.panel_conditional("input.show_activations"):
    @render.plot
    def plot_activation():
        plot_thing("activation", simulation_results(),
                   (None if input.show_prepop() else max_utility(gamble())))

with ui.panel_conditional("input.show_probs"):
    @render.plot
    def plot_probability():
        plot_thing("probability", simulation_results(),
                   (None if input.show_prepop() else max_utility(gamble())))
