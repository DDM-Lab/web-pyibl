from alhazen import IteratedExperiment
from datetime import datetime
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from pyibl import Agent, df_plot
from random import random
from shiny import reactive
from shiny.express import input, render, ui

# TODO:
#   allow dumping data as CSV?
#   allow downloading plots?
#   allow saving and restoring settings?
#   online doc?

DEFAULT_PREPOPULATED_MULTIPLIER = 1.2

help_text = """
# WebPyIBL

WebPyIBL is is a web hosted interactive demonstration of
[Instance-Based Learning](https://www.sciencedirect.com/science/article/abs/pii/S0364021303000314) (IBL),
implemented using the [PyIBL](http://pyibl.ddmlab.com) framework, and hosted by Carnegie
Mellon University's
[Dynamic Decision Making Laboratory](https://www.cmu.edu/dietrich/sds/ddmlab/).
It displays a simple iterated N-ary choice task, where a virtual user
repeatedly chooses between 2, 3 or 4 different options, each of
which returns a numeric payoff, the exact value of which may be probabilistically
determined, while trying to maximize that payoff. An IBL  model executes this task for a
fixed number of rounds, learning based solely on its experience. This model is typically
run for a number of different virtual participants, and the average results shown in
line plots.

Controls for altering a variety of parameters used by the task and model are shown on the
left portion of the screen, while the resulting plots are sown on the right. The controls
enable varying the number of options in the task, the possible payoffs and their
probabilities, the number of rounds and virtual participants, the IBL parameters used by
the model, and which plots are displayed.

This help text can be expanded to full screen by scrolling over its lower, right hand
corner and clicking the "Expand" button that pops up there. Or it can be hidden entirely
by using the switch in the upper right hand corner of the window.

In the panel on the left the first control is a popup allowing the choice of 2, 3 or 4
options. Immediately beneath it appear sets of controls for each of those options.
Blah, blah, blah....

<img src="under-construction.png" alt="alt text" width=256 height=211>

**This page still under construction.**

Lorem ipsum dolor sit amet, consectetur adipiscing elit. In in odio id nibh facilisis
suscipit. Sed vehicula rutrum leo lobortis sagittis. Praesent vel turpis risus. Nunc
molestie convallis metus sed tristique. Pellentesque semper lacinia tortor ac egestas. Sed
pellentesque urna metus, sit amet pellentesque tortor varius vitae. Maecenas elementum
porta scelerisque. Vivamus fermentum pharetra ornare.

Curabitur convallis ut est ut dignissim. Nullam rutrum ex vitae volutpat vehicula. Quisque
tellus enim, volutpat vitae iaculis non, malesuada vitae enim. Morbi quis porta libero.
Donec mi lorem, eleifend porta tempus vitae, consectetur eget dolor. Maecenas dignissim,
ligula ut imperdiet laoreet, elit elit volutpat lorem, nec semper metus ex at tortor. Orci
varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nulla
orci diam, pellentesque vitae nisl consequat, facilisis eleifend nunc. Vivamus lectus
lacus, pretium et suscipit a, hendrerit et augue.

Donec tellus magna, aliquet at viverra nec, sodales in purus. Sed quis lorem vel metus
commodo finibus id in metus. Nam posuere vulputate mi, eget ultricies odio. Curabitur quis
dui vel eros hendrerit tempus. Proin dignissim ullamcorper feugiat. Nunc mollis mauris sit
amet mollis vehicula. Quisque mauris ex, dictum et commodo ac, venenatis eget est.
Curabitur bibendum ipsum eget ipsum placerat sodales. Cras et sem in felis facilisis
iaculis. Fusce tellus augue, tristique at tristique eget, placerat non neque. Pellentesque
vel dignissim sapien. Integer pellentesque tortor vel turpis tempor faucibus. Duis
ultricies porttitor nisl. Etiam cursus felis non turpis placerat venenatis.

Etiam commodo ante vitae accumsan consectetur. Morbi iaculis orci id mollis semper. Nullam
congue mattis dui sit amet mollis. Integer nec dui eros. Donec in sodales nisi, non
blandit magna. Fusce accumsan aliquam felis vitae fringilla. Donec sollicitudin elit in
ipsum sagittis, at scelerisque tortor hendrerit. Suspendisse dui nibh, hendrerit et
sagittis non, imperdiet pretium felis. Donec quam risus, blandit at consequat laoreet,
eleifend vitae turpis. Praesent sapien magna, faucibus ut dapibus eu, blandit sed enim.

Sed blandit auctor sapien viverra aliquam. In sit amet consectetur lacus, ac euismod
lacus. Vivamus eu turpis ante. Donec risus dolor, suscipit vel vulputate eu, placerat et
justo. Curabitur in cursus ipsum, et viverra lacus. Pellentesque nec libero eu libero
consectetur tincidunt vitae faucibus leo. Duis accumsan justo sit amet nisl cursus
venenatis. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per
inceptos himenaeos. Nam lobortis, mauris vel accumsan viverra, magna nibh commodo justo,
sit amet sollicitudin elit nibh in urna.
"""


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

def _ev(low, prob, high):
    return (low * prob + high * (100 - prob)) / 100

with ui.sidebar(width=400):
    ui.tags.style("html {font-size: 65%;}")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ui.input_select("option_count", None, ("2 Options", "3 Options", "4 Options"), width="35%")
    # Grumble: like Jupyter Notebooks Shiny seems to encourage ugly, difficult to update
    #          cut and paste programming; it sure would be nice if there were some easy way
    #          to write an abstraction for the following many times repeated code, but
    #          I've not found one yet.
    with ui.card():
        with ui.layout_columns():
            ui.HTML("Option A")
            @render.text
            def A_ev():
                return f"Expected value: {_ev(input.A_low(), input.A_prob(), input.A_high()):.2f}"
        with ui.layout_columns(col_widths=(3, 6, 3)):
            ui.input_numeric("A_low", None, 3)
            ui.input_slider("A_prob", None, 0, 100, 100, step=1, post="%")
            ui.input_numeric("A_high", None, 3)
    with ui.card():
        with ui.layout_columns():
            ui.HTML("Option B")
            @render.text
            def B_ev():
                return f"Expected value: {_ev(input.B_low(), input.B_prob(), input.B_high()):.2f}"
        with ui.layout_columns(col_widths=(3, 6, 3)):
            ui.input_numeric("B_low", None, 0)
            ui.input_slider("B_prob", None, 0, 100, 75, post="%")
            ui.input_numeric("B_high", None, 4)
    with ui.panel_conditional("input.option_count === '3 Options' || input.option_count === '4 Options'"):
        with ui.card():
            with ui.layout_columns():
                ui.HTML("Option C")
                @render.text
                def C_ev():
                    return f"Expected value: {_ev(input.C_low(), input.C_prob(), input.C_high()):.2f}"
            with ui.layout_columns(col_widths=(3, 6, 3)):
                ui.input_numeric("C_low", None, 0)
                ui.input_slider("C_prob", None, 0, 100, 50, post="%")
                ui.input_numeric("C_high", None, 0)
    with ui.panel_conditional("input.option_count === '4 Options'"):
        with ui.card():
            with ui.layout_columns():
                ui.HTML("Option D")
                @render.text
                def D_ev():
                    return f"Expected value: {_ev(input.D_low(), input.D_prob(), input.D_high()):.2f}"
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
        ui.input_switch("show_probs", "Show probabilities of retrieval", True)
        ui.input_switch("show_activations", "Show activations", True)
        ui.input_switch("show_baselevel", "Show baselevel activations", True)
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
                ui.input_numeric("prepop_multiplier", None, DEFAULT_PREPOPULATED_MULTIPLIER,
                                 min=1.0, step=0.1)
    with ui.layout_columns(col_widths=(5, 6, 1)):
        ui.input_action_button("recompute", "Recompute")
        ui.HTML("&nbsp;")
        ui.input_dark_mode(mode="light")

with ui.layout_columns(col_widths=(10, 2)):
    with ui.panel_conditional("!input.hide_help"):
        with ui.card(full_screen=True, height="18ex"):
            ui.markdown(help_text)
    ui.input_switch("hide_help", "Hide", False)


def plot_thing(kind, df, title, max=False):
    df_plot(df, kind, title=title, xlabel="Round", show=False,
            max=(None if not max or input.show_prepop() else max_utility(gamble())))

@render.plot
def plot_choice():
    plot_thing("choice", simulation_results(), "Fraction making choice")

with ui.panel_conditional("input.show_bvs"):
    @render.plot
    def plot_blended_values():
        plot_thing("bv", simulation_results(), "Mean blended value")

with ui.panel_conditional("input.show_probs"):
    @render.plot
    def plot_probability():
        plot_thing("probability", simulation_results(), "Mean probability of retrieval", True)

with ui.panel_conditional("input.show_activations"):
    @render.plot
    def plot_activation():
        plot_thing("activation", simulation_results(), "Mean total activation", True)

with ui.panel_conditional("input.show_baselevel"):
    @render.plot
    def plot_baselevel():
        plot_thing("baselevel", simulation_results(), "Mean base level activation", True)
