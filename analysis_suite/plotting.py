"""
Plotting.py file for analysis_suite

Contains functions and classes for creating interactive plots

Adapted from https://stackoverflow.com/questions/31410043/hiding-lines-after-showing-a-pyplot-figure
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_interactive_chart(dataframe):
    """
    Plots a graph with an interactive legend where each entry can be hidden/shown

    Parameters
    ------
    dataframe : pandas dataframe
        pandas dataframe where each row is a bacteria and each column is a timepoint
    """
    # create figure
    fig, ax = plt.subplots()
    # itterate through the rows and plot each bacteria
    for ix, row in dataframe.iterrows():
        ax.plot(list(dataframe), row, label=r'$Well={}$'.format(ix))

    # set legend location, subplot location and figure title
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
              ncol=2, borderaxespad=0)
    fig.subplots_adjust(right=0.55)
    fig.suptitle('Right-click to hide all\nMiddle-click to show all',
                 va='top', size='large')

    # create interactive legend
    leg = interactive_legend()
    return # fig, ax, leg

def interactive_legend(ax=None):
    """
    Takes the legend and passes it into a class object which makes it interactive

    Parameters
    ------
    ax : matplotlib.axes
        The axes for the plot

    Returns
    ------
    InteractiveLegend : class instance
        an instance of an interactive legend which allows datapoints to be hidden/shown
    """
    if ax is None:
        ax = plt.gca()
    if ax.legend_ is None:
        ax.legend()

    return InteractiveLegend(ax.get_legend())

class InteractiveLegend(object):
    """
    # TODO:  read through this and add proper docstring/understand each function etc as taken
    from stackoverflow
    """
    def __init__(self, legend):
        self.legend = legend
        self.fig = legend.axes.figure

        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()

        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return

        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()

    def show(self):
        plt.show()
