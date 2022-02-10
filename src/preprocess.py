# coding: utf-8
# author: ztypl
# date:   2021/3/22

import json
import re
import copy
import nltk

import torch
from .tree_module import Tree

consts = ['3.14', '1', '2', '(1/2)']


def load_math23k(filename):
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""
    return data


def load_ape210k(filename):
    f = open(filename, encoding='utf-8')
    data = []
    for s in f:
        entry = json.loads(s)
        data.append(entry)
    return data


def filter_data(data):  # filter 纯计算题
    filter_count = 0
    new_data = []
    pattern1 = re.compile(r'^(计算|用简便方法计算|简便计算|解方程|解比例|巧算|列式计算|求解|简便运算|简算|求比值)：')
    pattern2 = re.compile(r"^x=\d+(\.\d*)?%?$")
    pattern4 = re.compile(r'(cm|dm|km|mm|hm|am|m)\*\*(2|3)')
    pattern5 = re.compile(r"(\d\.[^\d])|([^\d]\.[^\d])|([^\d]\.[\d])")
    for d in data:
        # count chinese char
        cc_count = 0
        for ch in d['original_text']:
            if u'\u4e00' <= ch <= u'\u9fa5':
                cc_count += 1
        if cc_count <= 5:
            filter_count += 1
        # conditional filter
        elif re.match(pattern1, d['original_text']):
            filter_count += 1
        elif re.match(pattern2, d['equation']):
            filter_count += 1
        elif '规律' in d['original_text'] or '()' in d['original_text']:
            filter_count += 1
        elif 'mim' in d['original_text']:
            d['original_text'] = d['original_text'].replace("mim", "min")
        elif re.search(pattern5, d['original_text']):  # handle dot
            filter_count += 1
        elif "**" in d['original_text']:  # handle **
            d['original_text'] = d['original_text'].replace("厘米**", "cm**").replace("米**", "m**")
            while "**" in d['original_text']:
                if re.search(pattern4, d['original_text']):
                    d['original_text'] = re.sub(pattern4, r"\1\2", d['original_text'])
                else:
                    filter_count += 1
                    break
            else:
                new_data.append(d)
        else:
            new_data.append(d)
    return new_data


def filter_invalid_char(data):
    dx = {}
    new_data = []
    symbols_chars = set("()/%=，．？.、：\"；…《》")  # +-*°∠<> filtered
    for d in data:
        for c in d['original_text']:
            if not (u'\u4e00' <= c <= u'\u9fa5'):
                dx[c] = dx.get(c, 0) + 1
    invalid_chars = set(dx.keys()) - set((chr(ord('a') + x) for x in range(26))) - set(
        [str(x) for x in range(10)]) - symbols_chars
    for d in data:
        if len(set(d['original_text']).intersection(invalid_chars)) == 0:
            new_data.append(d)
    return new_data


def tolower(data):
    pattern_upper = re.compile(r'[A-Z]')
    pattern_lower = re.compile(r'[a-z]')
    new_data = []
    for d in data:
        if re.search(pattern_upper, d['original_text']):  # 过滤大写字母，与小写字母同时出现则过滤（80个左右），否则替换为小写字母
            caps = set(re.findall(pattern_upper, d['original_text']))
            normals = set(re.findall(pattern_lower, d['original_text']))
            cap2nor = set()
            for x in caps:
                cap2nor.add(x.lower())
            inter = cap2nor.intersection(normals)
            if len(inter) == 0:  # 无 2lower 冲突
                d['original_text'] = d['original_text'].lower()
                new_data.append(d)
        else:
            new_data.append(d)
    return new_data


def clean_data(data):
    return filter_invalid_char(filter_data(tolower(data)))


def retokenize(data):
    tokenizer = nltk.RegexpTokenizer(r"""(?x)
        (?:mp[345]|mqw91|mn|hcf|he11o|miss)|             # 特殊单词
        (?:\(\d+/\d+\))|                            # 分数
        (?:\d+(?:\.\d+)?%?)|                        # 数
        (?:(?:ml|min|cm|dm|km|mm|hm|am|kg|mb|h|m)(?:2|3)?)|  # 单位
        (?:(?:(?<=[^a-z])|^)[a-z]+(?:(?=[^a-z])|$))|      # 孤立英文
        [^\da-z0-9]""")
    for i, d in enumerate(data):
        tokens = tokenizer.tokenize(d['original_text'])
        d['segmented_text'] = tokens
    return data


def filter_equation(data):
    new_data = []
    tokenizer = nltk.RegexpTokenizer(r"""(?x)
        (?:\(\d+/\d+\))|      # 数
        (?:\d+(?:\.\d+)?%?)|
        (?:\*\*|\+|\-|\*|/|\(|\))""")
    for d in data:
        equ = d['equation']
        if equ.startswith('x='):
            equ = equ[2:]
        tokens = tokenizer.tokenize(equ)
        if equ == "".join(tokens) and len(tokens) <= 25:
            d['tokenized_equation'] = tokens
            new_data.append(d)
    return new_data


def filter_number(data):
    new_data = []
    number_pattern = r"^(?:\(\d+/\d+\))|(?:\d+(?:\.\d+)?%?)$"
    for d in data:
        nums = []
        equ_nums = []
        tokens = d['segmented_text']
        for t in tokens:
            if re.match(number_pattern, t):
                nums.append(t)
        equ_tokens = d['tokenized_equation']
        for t in equ_tokens:
            if re.match(number_pattern, t):
                equ_nums.append(t)
        if len(nums) != len(set(nums)):
            pass
        elif len(set(equ_nums)) >= 2:
            if set(equ_nums) == set(nums):
                new_data.append(d)
            elif set(equ_nums) > set(nums) and (set(equ_nums) - set(nums)) <= set(consts):
                new_data.append(d)
    return new_data



def preprocess_number(data, postorder=True):
    new_data = []
    number_pattern = re.compile(r"^(?:\(\d+/\d+\))|(?:\d+(?:\.\d+)?%?)$")
    op_pri = {'+': 1, '-': 1, '*': 2, '/': 2, '**': 3}
    for d in data:
        if postorder:
            s1 = []
            s2 = []
            for i, t in enumerate(d['tokenized_equation']):
                if re.match(number_pattern, t):  # is a number
                    s2.append(t)
                elif t in op_pri.keys():  # is a operator
                    while len(s1) > 0 and s1[-1] in op_pri.keys() and op_pri[t] <= op_pri[s1[-1]]:
                        s2.append(s1.pop())
                    s1.append(t)
                elif t == '(':
                    s1.append(t)
                elif t == ')':
                    while len(s1) > 0 and s1[-1] != '(':
                        s2.append(s1.pop())
                    assert (s1[-1] == '(')
                    s1.pop()
            while len(s1) > 0:
                assert (s1[-1] != '(')
                s2.append(s1.pop())
        else:
            s2 = copy.copy(d['tokenized_equation'])
        num_dict = {}
        sx = copy.copy(d['segmented_text'])

        for i, t in enumerate(sx):
            if re.match(number_pattern, t):
                if t not in num_dict:
                    num_dict[t] = []
                num_dict[t].append(i)
        num_count = 1
        num_dict_equ = {}
        for i, t in enumerate(s2):
            if re.match(number_pattern, t):
                if t in num_dict:
                    if t not in num_dict_equ:
                        num_dict_equ[t] = num_count
                        num_count += 1
                    tmp = "N%d" % num_dict_equ[t]
                    s2[i] = tmp
                    for idx in num_dict[t]:
                        sx[idx] = tmp
        new_data.append({
            'input': s2,
            'output': sx
        })
    return new_data

class Tokens:
    SOS = '<SOS>'
    EOS = '<EOS>'
    PAD = '<PAD>'
    UNK = '<UNK>'

class Lang:
    def __init__(self, sos=False, eos=False, unk=False):
        special_tokens = [Tokens.PAD]
        if sos:
            special_tokens.append(Tokens.SOS)
        if eos:
            special_tokens.append(Tokens.EOS)
        if unk:
            special_tokens.append(Tokens.UNK)

        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        if special_tokens:
            self.index2word.extend(special_tokens)
            self.word2index = {w: i for i, w in enumerate(self.index2word)}
        self.n_words = len(self.index2word)

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def transfer_sentence(self, sentence, SOS=True, EOS=True):
        output = []
        if SOS:
            output.append(self.word2index[Tokens.SOS])
        for s in sentence:
            if s in self.word2index:
                output.append(self.word2index[s])
            elif Tokens.UNK in self.word2index:
                output.append(self.word2index[Tokens.UNK])
            else:
                raise KeyError("Unknown token %d and <UNK> not included in the language." % s)
        if EOS:
            output.append(self.word2index[Tokens.EOS])
        return output

    def transfer_sentence_into_tree(self, sentence, x_size, h_size, topic):
        stack = []
        tree = Tree(h_size, x_size)
        for s in sentence:
            i = self.word2index[s]
            if s in {'+', '-', '*', '/', '**'}:
                op1 = stack[-1]
                op2 = stack[-2]
                stack.pop()
                stack.pop()
                stack.append(tree.add_node_bottom_up(i, topic, [op1, op2]))
            else:
                stack.append(tree.add_node(i, topic))
        return tree


    def reverse_sentence(self, sentence):
        return [self.index2word[s] for s in sentence]


def generate_lang(data):
    input_lang = Lang(sos=False, eos=False)
    output_lang = Lang(sos=True, eos=True)
    for d in data:
        input_lang.add_sentence(d['input'])
        output_lang.add_sentence(d['output'])
    return input_lang, output_lang


def generate_dataset(data, input_lang: Lang, output_lang: Lang,
                     input_sos=False, input_eos=False,
                     output_sos=False, output_eos=True):
    dataset = []
    for d in data:
        sin = input_lang.transfer_sentence(d['input'], SOS=input_sos, EOS=input_eos)
        sout = output_lang.transfer_sentence(d['output'], SOS=output_sos, EOS=output_eos)
        dataset.append((sin, sout))
    return dataset

def generate_topic_dataset(data, input_lang: Lang, output_lang: Lang,
                     input_sos=False, input_eos=False,
                     output_sos=False, output_eos=True):
    dataset = []
    for d, topic in data:
        sin = input_lang.transfer_sentence(d['input'], SOS=input_sos, EOS=input_eos)
        sout = output_lang.transfer_sentence(d['output'], SOS=output_sos, EOS=output_eos)
        dataset.append((torch.LongTensor(sin), topic, torch.LongTensor(sout)))
    return dataset

def generate_tree_topic_dataset(data, input_lang: Lang, output_lang: Lang,
                     x_size, h_size,
                     output_sos=False, output_eos=True):
    dataset = []
    c = 0
    for d, topic in data:
        try:
            sin = input_lang.transfer_sentence_into_tree(d['input'], x_size, h_size, topic)
            sout = output_lang.transfer_sentence(d['output'], SOS=output_sos, EOS=output_eos)
            dataset.append((sin, torch.LongTensor(sout)))
        except IndexError:
            c += 1
    print(f"lose {c} records.")
    return dataset