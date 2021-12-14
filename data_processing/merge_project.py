import os
import argparse
import json
import pickle
from tqdm import tqdm

def get_project_data(path):
    """
    format of data:
    {
        "dir": name of this dir,
        "files": list of file contents, [{"name": file_name, "content": code}],
        "subdirs": list of subdirs, [data]
    }
    """
    data = {"dir": path.split("/")[-1], "files": [], "subdirs": []}
    for name in os.listdir(path):
        if os.path.islink(os.path.join(path, name)):
            continue
        elif os.path.isdir(os.path.join(path, name)):
            data["subdirs"].append(get_project_data(os.path.join(path, name)))
        elif os.path.isfile(os.path.join(path, name)):
            data["files"].append({"name": name, "content": open(os.path.join(path, name), encoding="latin1").read()})
    return data


        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge projects.')
    parser.add_argument("--data_path", type=str, default="/data2/liufang/datasets_javam/",
                        help="file path")
    parser.add_argument("--save_path", type=str, default="/data4/liufang/GTNM/raw/",
                        help="save path")
    args = parser.parse_args()

    train_data = []
    eval_data = []
    test_data = []

    projects = os.listdir(os.path.join(args.data_path, "training"))
    for project_name in tqdm(projects):
        train_data.append(get_project_data(os.path.join(args.data_path, "training", project_name)))
        eval_data.append(get_project_data(os.path.join(args.data_path, "evaluating", project_name)))
        test_data.append(get_project_data(os.path.join(args.data_path, "testing", project_name)))
    pickle.dump(train_data, open(args.save_path+'java-train.pkl', "wb"))
    pickle.dump(eval_data, open(args.save_path+'java-eval.pkl', "wb"))
    pickle.dump(test_data, open(args.save_path+'java-test.pkl', "wb"))
