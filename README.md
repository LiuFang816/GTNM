# GTNM
Implementation for the ICSE 2022 paper "Learning to Recommend Method Names with Global Context"

## data processing
### data source
- We used the same dataset as [MNire](https://dl.acm.org/doi/pdf/10.1145/3377811.3380926), [Code2vec](https://arxiv.org/pdf/1803.09473.pdf), and [DeepName](https://arxiv.org/pdf/2103.00269.pdf). All the repos are listed in `./data/repos.txt`
    - For in-project setting, we used the same setting as in MNire and code2vec to shuffle files in all the projects and split them into 1.7M training and 61K testing files.
    - For cross-project and low resource setting, we first split 9000/200/1022 projects for training/validation/testing. Then we sample 4000-projects from 9000-training-projects as the final training-set. The repos are listed in `./data/small_train_projects.txt`, `./data/eval_projects.txt`,  `./data/test_projects.txt`

### cross-project data processing   

1. run `merge_project.py` to save project information
    ```
    --data_path: project data dir
    --save_path: dir to save the project information data (`java-train.pkl, java-eval.pkl, java-test.pkl`)
    ```

2. run `processor.py` to get code schema and cross project information
    ```
    --input_file: # dir to save the project information data (For example: `data_path/java-train.pkl`)
    --schema_file # dir to save code schema information (For example: `data_path/java-train_schema.pkl`)
    --output_file # dir to save code schema and cross project information (For example: `data_path/java-train_all.pkl`)
    ```

3. run `extract_data.py` to save final pickle data
    ```
    --sub_vocab_file: vocabulary for subtokens in the source code 
    --doc_vocab_file: vocabulary for documentation of the methods
    --input_file_name: dir to save code schema and cross project information (For example: `data_path/java-train_all.pkl`)
    --output_file_name: dir prefix to save the final data for training and evaluation (For example: `data_path/train_subword`, following files will be saved:  `data_path/train_subword_body/doc/pro/tag.pkl`)
    ```

4. run `invoked_save.py` to save invoked mask for project context
    ```
    parameters:
    data_path: dir to save the final data for training and evaluation
    prefix: data prefix, for example: train_subword
    ```

## Model training and tesing
- parameters are configured in `hparams.py`

### Training

    run `train.py`

    python train.py --gpu gpu_id --pro True

### Testing
    run `test.py`

    python test.py --gpu gpu_id --pro True

