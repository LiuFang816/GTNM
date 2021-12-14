"""
get code schema and cross project info
"""
import os
from collections import Counter
from tqdm import tqdm
import pickle
import json
import re
import argparse
from tree_sitter import Parser, Language
from java_parser import JavaParser
import pathos

Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-java',
  ]
)

JAVAPARSER = Parser()
JAVAPARSER.set_language(Language("my-languages.so", "java"))
jp = JavaParser(parser=JAVAPARSER)

PARSER = None
sp_parser = None

class processor(object):
    def __init__(self, language):
        self.language = language
        global PARSER, sp_parser
        if language == "java":
            PARSER = JAVAPARSER
            sp_parser = jp

    def read_as_pkl(self, ifilename, ofilename):
        data = pickle.load(open(ifilename, "rb"))
        for i, project_data in enumerate(data):
            if i % 1000 == 0:
                print(i)
            yield project_data
        
        pickle.dump(data, open(ofilename, "wb"))
    
    def parse_schema(self, data):
        for file in data["files"]:
            try:
                sp_parser.update(file["content"])
                file["schema"] = sp_parser.schema
            except RecursionError:
                file["schema"] = {
                    "file_docstring": "",
                    "contexts": [],
                    "methods": [],
                    "classes": [],
                }       
        for subdir in data["subdirs"]:
            self.parse_schema(subdir)
    
    def process(self, ifilename, ofilename):
        """
        format of data:
        {
            "dir": name of this dir,
            "files": list of file contents, [{"name": file_name, "content": code, "schema": schema}],
            "subdirs": list of subdirs, [data]
        }
        """
        data = pickle.load(open(ifilename, "rb"))
        for project_data in tqdm(data[:4000]):
            self.parse_schema(project_data)
        pickle.dump(data[:4000], open(ofilename, "wb"))


    def search_dir(self, data, target, current=""):
        current = current+"."+data["dir"] if current else data["dir"]
        if current.endswith(target):
            return data
        for subdir in data["subdirs"]:
            res = self.search_dir(subdir, target, current)
            if res is not None:
                return res
        return None
    
    def cross_project_info_java(self, data, root_data=None, path=""):
        if root_data is None:
            root_data = data
        path = path+"."+data["dir"] if path else data["dir"]
        # print(path)
        for file in data["files"]:
            file["imports"] = []
            # print(file["name"])
            for import_stat in file["schema"]["contexts"]:
                if import_stat.startswith("import"):
                    # print(import_stat)
                    content = import_stat.split()[-1].rstrip(";").split(".")
                    if len(content) < 2:
                        continue
                    # import class
                    import_dir = ".".join(content[:-2])
                    if import_dir.startswith("java") or import_dir.startswith("com.intellij") or import_dir.startswith("android"):
                        continue
                    import_file = content[-2]
                    import_class = content[-1]
                    # print(import_dir)
                    res = self.search_dir(root_data, import_dir)
                    if res is not None:
                        for nfile in res["files"]:
                            if import_file == nfile["name"].split(".")[0]:
                                if import_class == "*":
                                    for cl in nfile["schema"]["classes"]:
                                        file["imports"].append(cl)
                                else:
                                    for cl in nfile["schema"]["classes"]:
                                        if cl["name"] == import_class:
                                            file["imports"].append(cl)
                                break
                    if import_class != "*":
                        # import file
                        import_dir = ".".join(content[:-1])
                        import_file = content[-1]
                        res = self.search_dir(root_data, import_dir)
                        if res is not None:
                            for nfile in res["files"]:
                                if import_file == nfile["name"].split(".")[0]:
                                    for cl in nfile["schema"]["classes"]:
                                        file["imports"].append(cl)
                                    break
        for subdir in data["subdirs"]:
            self.cross_project_info_java(subdir, root_data, path)
        # exit()

    
    def cross_project_process(self, ifilename, ofilename):
        data = pickle.load(open(ifilename, "rb"))
        for project_data in tqdm(data):
            self.cross_project_info_java(project_data)
        pickle.dump(data, open(ofilename, "wb"))


    def travel(self, data):
        for file in data["files"]:
            if len(file["imports"]) > 0:
                self.cnt += 1
        for subdir in data["subdirs"]:
            self.travel(subdir)

    def read_results(self, filename):
        data = pickle.load(open(filename, "rb"))
        self.cnt = 0
        for project_data in data:
            self.travel(project_data)
        print(self.cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processor.')
    parser.add_argument("--language", type=str, default="java",
                        help="language")
    parser.add_argument("--input_file", type=str, default="/data4/liufang/GTNM/raw/java-train.pkl",
                        help="the input file name")
    parser.add_argument("--schema_file", type=str, default="/data4/liufang/GTNM/raw/java-train_schema.pkl",
                        help="the output file name")
    parser.add_argument("--output_file", type=str, default="/data4/liufang/GTNM/raw/java-train_all.pkl",
                        help="the output file name")
    args = parser.parse_args()
    
    p = processor(args.language)
    p.process(args.input_file, args.schema_file)
    p.cross_project_process(args.schema_file, args.output_file)

