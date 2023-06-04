#!/usr/bin/env python
# Copyright (c) Marius Gheorghescu. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
import sys
import time
import platform
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.dates import num2date, MinuteLocator, SecondLocator, DateFormatter
from matplotlib.widgets import Button
from datetime import datetime, timedelta
from threading import Thread
from os import path


# from matplotlib.animation import FuncAnimation

refresh_interval = 66 # 66ms = 15fps

# controls the window size (and memory usage). 100k samples = 3 minutes
buffer_max_samples = 100000

# controls how many samples to display in the chart (and CPU usage). Ie 4k display should be ok with 2k samples
chart_max_samples = 2048

# how many samples to average (median) 
max_supersampling = 16

# set to true to compute median instead of average (less noise, more CPU)
median_filter = 0

# 
save_file = None
save_format = None

connected_device = "CurrentRanger"

class CRPlot:
    def __init__(self, sample_buffer = 100):
        self.port = '/dev/ttyACM0'
        self.baud = 9600
        self.thread = None
        self.stream_data = True
        self.pause_chart = False
        self.sample_count = 0
        self.animation_index = 0
        self.max_samples = sample_buffer
        self.data = collections.deque(maxlen=sample_buffer)
        self.timestamps = collections.deque(maxlen=sample_buffer)
        self.dataStartTS = None
        self.serialConnection = None
        self.framerate = 30
        self.version = 1



        
    def getData(self, frame, lines, legend, lastText):
        if (self.pause_chart or len(self.data) < 2):
            lastText.set_text('')
            return

        if not self.stream_data:
            self.ax.set_title('<Disconnected>', color="red")
            lastText.set_text('')
            return

        dt = datetime.now() - self.dataStartTS

        # capped at buffer_max_samples
        sample_set_size = len(self.data)

        timestamps = []
        samples = [] #np.arange(chart_max_samples, dtype="float64")

        subsamples = max(1, min(max_supersampling, int(sample_set_size/chart_max_samples)))
        
        # Sub-sampling for longer window views without the redraw perf impact
        for i in range(0, chart_max_samples):
            sample_index = int(sample_set_size*i/chart_max_samples)
            timestamps.append(self.timestamps[sample_index])
            supersample = np.array([self.data[i] for i in range(sample_index, sample_index+subsamples)])
            samples.append(np.median(supersample) if median_filter else np.average(supersample))

        self.ax.set_xlim(timestamps[0], timestamps[-1])

        # some machines max out at 100fps, so this should react in 0.5-5 seconds to actual speed
        sps_samples = min(512, sample_set_size);
        dt_sps = (np.datetime64(datetime.now()) - self.timestamps[-sps_samples])/np.timedelta64(1, 's');

        # if more than 1 second since last sample, automatically set SPS to 0 so we don't have until it slowly decays to 0
        sps = sps_samples/dt_sps if ((np.datetime64(datetime.now()) - self.timestamps[-1])/np.timedelta64(1, 's')) < 1 else 0.0
        lastText.set_text('{:.1f} SPS'.format(sps))
        if sps > 500:
            lastText.set_color("white")
        elif sps > 100:
            lastText.set_color("yellow")
        else:
            lastText.set_color("red")


        # logging.debug("Drawing chart: range {}@{} .. {}@{}".format(samples[0], timestamps[0], samples[-1], timestamps[-1]))
        lines.set_data(timestamps, samples)
        self.ax.legend(labels=['Last: {}\nAvg: {}'.format( self.textAmp(samples[-1]), self.textAmp(sum(samples)/len(samples)))])


    def chartSetup(self, refresh_interval=100):
        plt.style.use('dark_background')
        fig = plt.figure(num=f"Current Viewer {self.version}", figsize=(10, 6))
        self.ax = plt.axes()
        ax = self.ax

        ax.set_title(f"Streaming: {connected_device}", color="white")

        # fig.text (0.2, 0.88, f"CurrentViewer {self.version}", color="yellow",  verticalalignment='bottom', horizontalalignment='center', fontsize=9, alpha=0.7)
        # fig.text (0.89, 0.0, f"github.com/MGX3D/CurrentViewer", color="white",  verticalalignment='bottom', horizontalalignment='center', fontsize=9, alpha=0.5)

        ax.set_ylabel("Current draw (Amps)")
        ax.set_yscale("log", nonpositive='clip')
        ax.set_ylim(1e-10, 1e1)
        plt.yticks([1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0], ['1nA', '10nA', '100nA', '1\u00B5A', '10\u00B5A', '100\u00B5A', '1mA', '10mA', '100mA', '1A'], rotation=0)
        ax.grid(axis="y", which="both", color="yellow", alpha=.3, linewidth=.5)

        ax.set_xlabel("Time")
        plt.xticks(rotation=20)
        ax.set_xlim(datetime.now(), datetime.now() + timedelta(seconds=10))
        ax.grid(axis="x", color="green", alpha=.4, linewidth=2, linestyle=":")

        #ax.xaxis.set_major_locator(SecondLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

        def on_xlims_change(event_ax):
            # logging.debug("Interactive zoom: {} .. {}".format(num2date(event_ax.get_xlim()[0]), num2date(event_ax.get_xlim()[1])))

            chart_len = num2date(event_ax.get_xlim()[1]) - num2date(event_ax.get_xlim()[0])

            if chart_len.total_seconds() < 5:
                self.ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S.%f'))
            else:
                self.ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
                self.ax.xaxis.set_minor_formatter(DateFormatter('%H:%M:%S.%f'))

        ax.callbacks.connect('xlim_changed', on_xlims_change)

        lines = ax.plot([], [], label="Current")[0]

        lastText = ax.text(0.50, 0.95, '', transform=ax.transAxes)
        statusText = ax.text(0.50, 0.50, '', transform=ax.transAxes)
        self.anim = animation.FuncAnimation(fig, self.getData, fargs=(lines, plt.legend(), lastText), interval=refresh_interval)

        plt.legend(loc="upper right", framealpha=0.5)

        apause = plt.axes([0.91, 0.15, 0.08, 0.07])
        self.bpause = Button(apause, label='Pause', color='0.2', hovercolor='0.1')
        self.bpause.on_clicked(self.pauseRefresh)
        self.bpause.label.set_color('yellow')

        aanimation = plt.axes([0.91, 0.25, 0.08, 0.07])
        self.bsave = Button(aanimation, 'GIF', color='0.2', hovercolor='0.1')
        # self.bsave.on_clicked(self.saveAnimation)
        # self.bsave.label.set_color('yellow')

        crs = mplcursors.cursor(ax, hover=True)
        @crs.connect("add")
        def _(sel):
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="yellow", alpha=.4)
            sel.annotation.set_text(self.textAmp(sel.target[1]))

        self.framerate = 1000/refresh_interval
        plt.gcf().autofmt_xdate()
        plt.show()

    def pauseRefresh(self, state):
        self.pause_chart = not self.pause_chart
        if self.pause_chart:
            self.ax.set_title('<Paused>', color="yellow")
            self.bpause.label.set_text('Resume')
        else:
            self.ax.set_title(f"Streaming: {connected_device}", color="white")
            self.bpause.label.set_text('Pause')
 
    def textAmp(self, amp):
        if (abs(amp) > 1.0):
            return "{:.3f} A".format(amp)
        if (abs(amp) > 0.001):
            return "{:.2f} mA".format(amp*1000)
        if (abs(amp) > 0.000001):
            return "{:.1f} \u00B5A".format(amp*1000*1000)
        return "{:.1f} nA".format(amp*1000*1000*1000)


# def animation_frame(i):
#     x_data.append(augmented.index[i])
#     y_data.append(ylikes[0][i])

#     line.set_xdata(x_data)
#     line.set_ydata(y_data)
#     return line,

#     plt.cla()
#     plt.tight_layout()
#     XN, YN = augment(x, y, 3)
#     augmented = pd.DataFrame(YN, XN)

#     ylikes = augmented[0].reset_index()  # Index reset to avoid key error

def main():
    # Read the CSV file
    data = pd.read_csv("simple_6_flash_led_data.log",delimiter=',')  #Timestamp, Amps
    data
    print(data.head(5))

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Initialize the line plot
    line, = ax.plot(data['Timestamp'], data[' Amps'], lw=2)

    # Set the axis labels
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('microAmps')

    # # Function to update the plot with each animation frame
    # def update(frame):
    #     # Get the data for the current frame
    #     x = data['Timestamp'][:frame]
    #     y = data[' Amps'][:frame]
        
    #     # Update the line plot
    #     line.set_data(x, y)
        
    #     # Set the x-axis limits to accommodate all data points
    #     ax.set_xlim(data['Timestamp'].min(), data['Timestamp'].max())
        
    #     return line,

    # # Create the animation
    # animation = FuncAnimation(fig, update, frames=len(data), interval=100, blit=True)

    # Show the plot
    plt.show()





if __name__ == '__main__':
  main()
