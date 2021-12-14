"""
Tokenize and normalize source code
"""

import re
import argparse
from tree_sitter import Parser, Language
import pathos

LIT_TYPES = {
    "java": [
        ["string_literal"],
        [
            "decimal_integer_literal",
            "hex_integer_literal",
            "binary_integer_literal",
            "decimal_floating_point_literal",
        ]
    ],
    "python": [
        ["string"],
        ["integer", "float"]
    ]
}

class Normalizer(object):
    def __init__(self, language):
        self.language = language
        self.parser = Parser()
        self.parser.set_language(Language("/data4/liufang/GTNM/preprocess/my-languages.so", language))
        self.lit_types = LIT_TYPES[language]

    def get_tokens(self, node, tokens, types):
        """
        Get all tokens from a TreeSitter like root node recursively.

        String-type node will be seen as one token.
        """
        if len(node.children) == 0:
            tokens.append([node.start_point, node.end_point])
            types.append(str(node.type))
            return
        if (
            str(node.type) not in ["concatenated_string", "string_array", "chained_string"]
            and "string" in str(node.type)
            or "char" in str(node.type)
        ):
            tokens.append([node.children[0].start_point, node.children[-1].end_point])
            types.append(str(node.type))
            return
        for child in node.children:
            self.get_tokens(child, tokens, types)


    def file_tokenizer(self, code):
        """
        Tokenize a source code snippet. (File, method or anything can be parsed by tree-sitter is ok)
        """
        try:
            tree = PARSER.parse(bytes(code, "utf8"))
            root = tree.root_node
            tokens = []
            types = []
            self.get_tokens(root, tokens, types)
            _, tokens, _ = self._file_tokenizer(code, tokens, types, False)
            return tokens
        except Exception:
            return []


    def _file_tokenizer(self, code, positions, types, keep_newline=True):
        """
        Tokenize a file from token positions and their types. Return positions, code tokens and types.

        Returned positions and types are not exact same as the original. '\\n' with no position and type 'new_line' is added.
        """
        code = bytes(code, "utf8")
        code = code.split(b"\n")
        prev_line = 0
        ret_pos = []
        ret_code = []
        ret_type = []
        for i, token in enumerate(positions):
            sp = token[0]
            ep = token[1]
            if sp[0] != prev_line and keep_newline:
                ret_pos.append([])
                ret_code.append("\n")
                ret_type.append("new_line")
            prev_line = ep[0]
            if sp[0] == ep[0]:
                ret_pos.append(token)
                ret_code.append(code[sp[0]][sp[1] : ep[1]].decode("utf-8"))
                ret_type.append(types[i])
            else:
                out = code[sp[0]][sp[1] :]
                for lineid in range(sp[0] + 1, ep[0]):
                    out += code[lineid]
                out += code[ep[0]][: ep[1]]
                ret_pos.append(token)
                ret_code.append(out.decode("utf-8"))
                ret_type.append(types[i])

        return ret_pos, ret_code, ret_type
    
    def normalize(self, code):
        tree = self.parser.parse(bytes(code, "utf8"))
        root = tree.root_node
        tokens = []
        types = []
        try:
            self.get_tokens(root, tokens, types)
        except RecursionError:
            return ""
        poss, tokens, types = self._file_tokenizer(code, tokens, types)
        norm_code = self.norm_untokenize(
            poss,
            tokens,
            types,
            self.lit_types,
            "remove",
        )
        return norm_code

    def norm_untokenize(
        self,
        poses,
        tokens,
        types,
        lit_types=[[],[]],
        comment="remove",
    ):
        code_string = []
        prev_sp = None
        prev_ep = None
        for pos, token, tp in zip(poses, tokens, types):
            if tp == "new_line" or tp == "\n":
                code_string += ["\n"]
                continue
            if "comment" in tp:
                if comment == "normalize":
                    code_string += ["#<COMMENT>"]
                elif comment == "keep":
                    code_string += [token]
                continue
            sp = pos[0]
            ep = pos[1]
            add_token = token
            # special token maps can't convert non-literal tokens
            if tp in lit_types[0]:
                pass
                # str_quote_options = ["'''", '"""', "'", '"']
                # start_quote = ""
                # end_quote = ""
                # qualifier_regex = r"^[a-z]+"
                # qualifier_match = re.search(qualifier_regex, token)
                # # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
                # qualifier = "" if not qualifier_match else qualifier_match[0]
                # # token string without qualifiers
                # token_string = re.sub(qualifier_regex, "", token)
                # # string literal without quotes
                # str_lit = token_string
                # for q in str_quote_options:
                #     if token_string.startswith(q):
                #         start_quote = q
                #         str_lit = str_lit[len(q) :]
                #         if token_string.endswith(q):
                #             end_quote = q
                #             str_lit = str_lit[: -len(q)]
                #         break
                # if start_quote in str_quote_options[:2]:
                #     add_token = ""
                # else:
                #     add_token = (
                #         f"{qualifier}{start_quote}{str_lit}{end_quote}"
                #         if len(str_lit) < 15 and "\n" not in str_lit and "</s>" not in str_lit and "<s>" not in str_lit and "<pad>" not in str_lit and "<EOL>" not in str_lit
                #         else f"{qualifier}{start_quote}{end_quote}"
                #     )
            elif tp in lit_types[1]:
                add_token = "0" if len(token) >= 10 else token

            code_string += [add_token]

            # if prev_sp is None or (sp[0] == prev_ep[0] and sp[1] == prev_ep[1]):
            #     code_string += add_token
            # elif sp[0] == prev_ep[0]:
            #     for i in range(prev_ep[1], sp[1]):
            #         code_string += " "
            #     code_string += add_token
            # else:
            #     code_string += "\n"
            #     for i in range(sp[1]):
            #         code_string += " "
            #     code_string += add_token
            prev_sp, prev_ep = sp, ep
        processed_code = " ".join(code_string).lstrip()
        return re.sub(re.compile("\s*\n"), "\n", processed_code)


# normalizer = Normalizer("java")
# print(normalizer.normalize("""\
# import argparse
# import logging
# import pickle

# class localContext(object):
#     def __init__(self, local_context_size, context_size, vocab_file, main_patterns, include_docstring=False, expr_max_len=1024, expr_max_num=30, custom_eol="<endofwhatthe funckline>", custom_eot="<endoftext>"):
#         self.local_context_size = local_context_size
#         self.context_size = context_size

#     def read_as_pkl(self, filename):
#         data = pickle.load(open(filename, "rb"))
#         for i, project_data in enumerate(data):
#             if i % 100 == 0:
#                 print(i)
#             for datax in self.travel_data(project_data):
#                 yield datax
# """))


