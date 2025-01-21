import kagglehub
import shutil

if __name__ == '__main__':
    target_dir = './data'
    path = kagglehub.dataset_download("selfishgene/historical-hourly-weather-data")
    print("Path to dataset files:", path)
    shutil.copytree(path, target_dir, dirs_exist_ok=True)  # Fine
    print("Copying dataset to :", target_dir)
    

