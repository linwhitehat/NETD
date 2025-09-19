#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @author:         lxj
# @description:    A unified script to generate the NETD (Dynamic Non-I.I.D. 
#                  Encrypted Traffic Dataset) based on the ISCX-VPN dataset.
#                  This script implements both Proportional Bias (for NETD-1, NETD-2)
#                  and Compositional Bias (for NETD-3, NETD-4) strategies as
#                  described in the paper "Respond to Change with Constancy...".

import os
import random
import shutil
import math

def create_proportional_bias_dataset(base_path: str, output_path: str, dominant_ratio: int):
    """
    Constructs an O.O.D. dataset by creating a proportional bias between a randomly
    selected "dominant" application and other "minor" applications within each service class.

    This method is used for generating NETD-1 and NETD-2.

    Args:
        base_path (str): The path to the source ISCX-VPN dataset, containing service class folders.
        output_path (str): The path where the generated dataset will be saved.
        dominant_ratio (int): The ratio of dominant to minor samples. For a 1:3 dominant-to-minor
                               sample count, this value should be 3. For 3:1, it should be 1/3.
    """
    print(f"--- Creating Proportional Bias Dataset at {output_path} ---")
    
    # Total samples to draw for the training set's dominant component
    total_train_dominant_samples = 400
    # Samples to draw for the test set (fixed 1:1 ratio)
    total_test_samples = 100

    service_labels = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for label in service_labels:
        print(f"Processing service class: {label}...")
        label_path = os.path.join(base_path, label)
        applications = [app for app in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, app))]

        if not applications:
            continue

        dominant_app = random.choice(applications)
        minor_apps = [app for app in applications if app != dominant_app]

        # Aggregate all file paths for dominant and minor applications
        dominant_files = [os.path.join(root, file) for root, _, files in os.walk(os.path.join(label_path, dominant_app)) for file in files]
        minor_files = [os.path.join(root, file) for app in minor_apps for root, _, files in os.walk(os.path.join(label_path, app)) for file in files]

        # Handle cases with only one application or insufficient minor files
        if len(applications) == 1:
            minor_files = dominant_files.copy()

        if not minor_files:
            print(f"Warning: No minor files for label {label}. Using dominant files as minor.")
            minor_files = dominant_files.copy()

        # --- Sample for Training Set ---
        train_dominant_samples = random.sample(dominant_files, min(total_train_dominant_samples, len(dominant_files)))
        
        num_minor_samples = int(total_train_dominant_samples / dominant_ratio)
        train_minor_samples = random.sample(minor_files, min(num_minor_samples, len(minor_files)))
        
        # --- Sample for Test Set (ensuring no overlap with training set) ---
        remaining_dominant = list(set(dominant_files) - set(train_dominant_samples))
        remaining_minor = list(set(minor_files) - set(train_minor_samples))

        test_dominant_samples = random.sample(remaining_dominant, min(total_test_samples, len(remaining_dominant)))
        test_minor_samples = random.sample(remaining_minor, min(total_test_samples, len(remaining_minor)))

        # --- Save the sampled files ---
        for split, samples in [("train", train_dominant_samples + train_minor_samples), ("test", test_dominant_samples + test_minor_samples)]:
            for file_path in samples:
                dest_dir = os.path.join(output_path, split, label)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(file_path, dest_dir)
    
    print(f"--- Proportional Bias Dataset created successfully. ---\n")


def create_compositional_bias_dataset(base_path: str, output_path: str, service_components: dict, training_ratio: float):
    """
    Constructs an O.O.D. dataset by creating a compositional bias, where the training
    set contains only a subset of applications for each service class, while the test set
    contains all of them.

    This method is used for generating NETD-3 and NETD-4.

    Args:
        base_path (str): The path to the source ISCX-VPN dataset.
        output_path (str): The path where the generated dataset will be saved.
        service_components (dict): A mapping of service names to a list of their application folder names.
        training_ratio (float): The percentage of applications to include in the training set (e.g., 0.8 for 80%).
    """
    print(f"--- Creating Compositional Bias Dataset at {output_path} ---")

    for service, apps in service_components.items():
        print(f"Processing service class: {service}...")
        
        # --- Determine Training Set Composition ---
        num_apps_for_training = math.ceil(len(apps) * training_ratio)
        training_apps = random.sample(apps, num_apps_for_training)
        
        print(f"  Training with {len(training_apps)}/{len(apps)} apps: {training_apps}")

        # --- Copy files for Training Set ---
        for app in training_apps:
            app_path = os.path.join(base_path, service, app)
            if os.path.exists(app_path):
                for root, _, files in os.walk(app_path):
                    for file in files:
                        dest_dir = os.path.join(output_path, "train", service)
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy(os.path.join(root, file), dest_dir)

        # --- Copy files for Test Set (all applications) ---
        for app in apps:
            app_path = os.path.join(base_path, service, app)
            if os.path.exists(app_path):
                for root, _, files in os.walk(app_path):
                    for file in files:
                        dest_dir = os.path.join(output_path, "test", service)
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy(os.path.join(root, file), dest_dir)

    print(f"--- Compositional Bias Dataset created successfully. ---\n")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    # !!! PLEASE UPDATE THESE PATHS !!!
    # Path to the extracted ISCX-VPN dataset, which should have subdirectories for each service 
    # (e.g., .../iscx_vpn_dataset/Chat/, .../iscx_vpn_dataset/Email/)
    ISCX_BASE_PATH = "path/to/your/iscx_vpn_dataset" 
    
    # Path where the generated NETD datasets will be stored.
    NETD_OUTPUT_PATH = "path/to/save/NETD"
    
    # Application mapping based on the paper  and ISCX-VPN dataset structure.
    # Note: Folder names must match those in your ISCX_BASE_PATH.
    SERVICE_APP_MAPPING = {
        'Chat': ['AIMchat', 'facebookchat', 'hangoutschat', 'icqchat', 'skypechat'],
        'Email': ['gmail', 'imap', 'pop', 'smtp'],
        'File Transfer': ['ftps', 'sftp', 'skypefile'],
        'P2P': ['bittorrent'],
        'Streaming': ['vimeo', 'youtube'],
        'VoIP': ['facebookvoip', 'hangoutsvoip', 'skypevoip']
    }

    if not os.path.isdir(ISCX_BASE_PATH) or not os.path.isdir(NETD_OUTPUT_PATH):
        print("Error: Please update ISCX_BASE_PATH and NETD_OUTPUT_PATH to valid directories.")
    else:
        # --- GENERATE DATASETS ---

        # Generate NETD-1: Proportional bias with a 1:3 dominant-to-minor ratio[cite: 610].
        # N_minor = N_dominant / ratio. For N_dom:N_min=1:3, ratio=3.
        create_proportional_bias_dataset(
            base_path=ISCX_BASE_PATH,
            output_path=os.path.join(NETD_OUTPUT_PATH, "NETD-1"),
            dominant_ratio=3 
        )

        # Generate NETD-2: Proportional bias with a 3:1 dominant-to-minor ratio[cite: 611].
        # For N_dom:N_min=3:1, ratio=1/3.
        create_proportional_bias_dataset(
            base_path=ISCX_BASE_PATH,
            output_path=os.path.join(NETD_OUTPUT_PATH, "NETD-2"),
            dominant_ratio=(1/3)
        )

        # Generate NETD-3: Compositional bias with 80% of apps in the training set[cite: 612, 613].
        create_compositional_bias_dataset(
            base_path=ISCX_BASE_PATH,
            output_path=os.path.join(NETD_OUTPUT_PATH, "NETD-3"),
            service_components=SERVICE_APP_MAPPING,
            training_ratio=0.8
        )

        # Generate NETD-4: Compositional bias with 20% of apps in the training set[cite: 615].
        create_compositional_bias_dataset(
            base_path=ISCX_BASE_PATH,
            output_path=os.path.join(NETD_OUTPUT_PATH, "NETD-4"),
            service_components=SERVICE_APP_MAPPING,
            training_ratio=0.2
        )
