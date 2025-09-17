# YOLOv11 Rice Detection Training Pipeline (Google Colab)

This notebook provides a complete, step-by-step pipeline for training a YOLOv11 object detection model on a rice dataset using Google Colab. It covers dataset transfer from Nextcloud, data preparation, class distribution analysis, configuration, training, validation, and saving results to Google Drive.

---

## Features

- **Google Drive Integration:** Automatically mounts Google Drive for saving results and persistent storage.
- **GPU Check:** Verifies GPU availability for efficient training.
- **Automated Library Installation:** Installs all required Python packages (Ultralytics, pandas, etc.).
- **Secure Dataset Transfer:** Uses rclone and Colab secrets to securely sync datasets from Nextcloud to Colab.
- **Flexible Dataset Paths:** Easily switch between test and production datasets by changing a single variable.
- **Class Distribution Analysis:** Analyzes and displays the number of instances per class in both training and validation sets.
- **Automatic YAML Config Generation:** Reads class names and generates a `data.yaml` file for YOLO training.
- **YOLOv11 Training:** Runs a baseline YOLOv11 training session with customizable parameters.
- **Validation:** Optionally runs validation on the trained model.
- **Results Backup:** Saves the entire `runs` folder (training outputs) to Google Drive for future access.

---

## Usage Instructions

### 1. Setup & Prerequisites
- Open this notebook in Google Colab.
- Ensure you have access to the required Nextcloud and Google Drive accounts.
- Store your Nextcloud credentials (hostname, username, password) in Colab's `userdata` secrets.
- Ensure you're using T4 GPU as your hardware accelerator in Colab.

### 2. Steps Overview

1. **Mount Google Drive**
   - The notebook will prompt you to authorize access to your Google Drive.

2. **Check GPU**
   - Verifies that a GPU is available for training.

3. **Install Required Libraries**
   - Installs Ultralytics (YOLOv11), pandas, and other dependencies.

4. **Transfer Dataset from Nextcloud**
   - Installs and configures rclone using your Nextcloud credentials.
   - Syncs the dataset (images and labels) from Nextcloud to Colab.
   - Supports both test and production datasets.

5. **Analyze Class Distribution**
   - Reads all label files and counts the number of instances per class for both training and validation sets.
   - Displays the results in a table.

6. **Generate data.yaml**
   - Reads class names from the provided file and generates a YOLO-compatible YAML configuration file.

7. **Train YOLOv11 Model**
   - Runs the training session (default: 50 epochs, 640x640 image size, 50% of data).
   - Outputs results to the `runs/padi_detection/full_training` directory.

8. **Validate Model (Optional)**
   - Runs validation on the trained model and specified classes.

9. **Save Results to Google Drive**
   - Copies the entire `runs` folder to a dedicated directory in your Google Drive for backup and sharing.

---

## Customization

- **Dataset Paths:**
  - Change the `remote_base_rclone` and `local_base` variables to switch between test and production datasets.
- **Training Parameters:**
  - Modify the YOLO training command to adjust epochs, image size, model type, or other hyperparameters.
- **Class Selection:**
  - Update the `classes` argument in the validation cell to focus on specific classes.

---

## Troubleshooting

- **Google Drive Mount Issues:**
  - Ensure you are logged into the correct Google account and have granted Colab the necessary permissions.
- **Nextcloud Sync Errors:**
  - Double-check your Nextcloud credentials in Colab's `userdata` secrets.
  - Ensure rclone is properly installed and configured.
- **Missing Classes or Labels:**
  - Verify that your dataset structure matches the expected format (images and labels in separate folders, class names file present).
- **Training Errors:**
  - Ensure all dependencies are installed and GPU is available.
  - Check the `data.yaml` file for correct paths and class names.

---

## Folder Structure

- `/content/dataset_padi/` or `/content/dataset_padi_test/`: Local dataset directory in Colab.
- `/content/drive/MyDrive/YOLOv11_Padi_Runs/`: Google Drive directory for saving training results.
- `runs/`: YOLOv11 output directory (training logs, weights, etc.).

---

## Credits

- Built using [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- Dataset transfer powered by [rclone](https://rclone.org/) and Nextcloud WebDAV.