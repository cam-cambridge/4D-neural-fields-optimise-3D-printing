import torch
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
import csv 
import os 

class Logger:
    def __init__(self, batch, v_num):
        self.metrics = {}
        self.batch = batch
        self.csv_files = {}
        self.v_num= v_num

        if not os.path.exists(f"logs/train/lightning_logs/version_{self.v_num}/metrics"):
            os.makedirs(f"logs/train/lightning_logs/version_{self.v_num}/metrics")

    def log(self, key, value):
        if key not in self.metrics:
            self.metrics[key] = []
            csv_file_path = f"logs/train/lightning_logs/version_{self.v_num}/metrics/{key}.csv"
            self.csv_files[key] = csv_file_path
            if not os.path.exists(csv_file_path):
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Value'])

        # Append the value to the metric list
        self.metrics[key].append(value)

        # Append the value to the CSV file
        with open(self.csv_files[key], mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([value])

    def get_metrics(self):
        return self.metrics

    def report_running_mean(self, plot):
        
        report = ">|"
        for key, values in self.metrics.items():
            mean_value = sum(values[-self.batch:]) / len(values[-self.batch:])
            report += f"|{key}: {colored(f'{mean_value:.4f}','blue')}|"

        if plot:
            self.plot_metrics()

    def plot_metrics(self):
        plt.figure(figsize=(10, 6),dpi=250)
        
        for key, values in self.metrics.items():
            color = "black"
            plt.plot(values, alpha=0.4, color=color)

        plt.xlabel('Batch')
        plt.yscale('log')
        plt.ylabel('Value')
        plt.title('Training Metrics Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"logs/train/lightning_logs/version_{self.v_num}/loss_plot.jpg")
        plt.close()